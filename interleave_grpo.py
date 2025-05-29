import torch
import torch.nn.functional as F
from torch import Tensor
import contextlib
import time
import datetime
import os
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
from lightning.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from lightning.fabric.strategies import DeepSpeedStrategy

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import torch
import argparse

# Rich imports for beautiful terminal output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.text import Text
from rich.tree import Tree
from rich.layout import Layout
from rich.align import Align
from rich import box
from rich.padding import Padding

# Create a global rich console
console = Console()

torch.set_float32_matmul_precision('medium')



@dataclass
class TrainingConfig:
    # Model and tokenizer configuration
    model_name: str = "joey00072/Llama-3.2-1B-Instruct-cold-start-ft2"
    tokenizer_name: str = "unsloth/meta-Llama-3.1-8B-Instruct"
    system_prompt: str = (
        "Respond in following format:<thinking>{step by step reasoning}</thinking>"
        "<answer>{number}</answer>"
    )
    
    # Training hyperparameters
    group_size: int = 8
    batch_size: int = 1
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_iterations: int = 1000
    max_new_tokens: int = 512
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    beta: float = 0
    epsilon_low: float = 0.1
    epsilon_high: float = 0.1
    
    # Logging configuration
    log_wandb: bool = True
    project_name: str = "nanoGRPO-DEV"
    
    # Device and dtype configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # LoRA configuration
    lora_r: int = 2
    lora_alpha: int = 16
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"])
    
    # Optimizer configuration
    optimizer_betas: tuple = (0.9, 0.95)
    
    # Checkpointing - ALWAYS enabled by default
    checkpoint_dir: str = "checkpoints"
    save_every: int = 4  # Save checkpoint every N iterations for safety
    resume_from: Optional[str] = None  # Specific checkpoint path to resume from (overrides auto-detection)
    # Note: Checkpoint resuming is ALWAYS enabled - the system will automatically 
    # look for and resume from the latest checkpoint in checkpoint_dir
    
    # Torch compile options
    torch_compile_options: Dict[str, Any] = field(default_factory=lambda: {
        "epilogue_fusion": True,
        "max_autotune": False,
        "shape_padding": True,
        "trace.enabled": False,
        "triton.cudagraphs": False,
    }) 

@torch.compile(dynamic=True, fullgraph=True)
def selective_log_softmax(logits: Tensor, index: Tensor) -> Tensor:
    per_token_logps = []
    for row_logits, row_labels in zip(logits, index):
        row_logps = F.log_softmax(row_logits, dim=-1)
        row_per_token_logps = row_logps.gather(
            dim=-1, index=row_labels.unsqueeze(-1)
        ).squeeze(-1)
        per_token_logps.append(row_per_token_logps)
    return torch.stack(per_token_logps)

def get_per_token_logps(model: AutoModelForCausalLM, input_ids: Tensor) -> Tensor:
    logits = model(input_ids=input_ids).logits
    logits = logits[:, :-1, :]
    input_ids = input_ids[:, 1:]
    return selective_log_softmax(logits, input_ids)

def compute_loss(
    model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    inputs: Tensor,
    old_logps: Tensor,
    rewards: Tensor,
    mean_r: Tensor,
    std_r: Tensor,
    loss_mask: Tensor,
    config: TrainingConfig,
    fabric_instance: Optional[Fabric] = None
) -> Tensor:
    policy_logps = get_per_token_logps(model, inputs)
    
    if config.beta > 0:
        with (ref_model.disable_adapter() if hasattr(ref_model, 'disable_adapter') else contextlib.nullcontext()):
            ref_logps = get_per_token_logps(ref_model, inputs)

    advantage = (rewards - mean_r) / (std_r + 1e-6)
    advantage = advantage.view(-1, 1)

    ratio = torch.exp(policy_logps - old_logps.detach())
    loss_clip = torch.clamp(ratio, 1 - config.epsilon_low, 1 + config.epsilon_high) * advantage
    loss = -torch.min(ratio * advantage, loss_clip)
    loss = (loss * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-6)
    loss = loss.mean()

    if config.beta > 0:
        log_ratio = ref_logps - policy_logps
        kld = torch.exp(log_ratio) - log_ratio - 1
        kld_term = (kld * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-6)
        loss = loss + config.beta * kld_term.mean()
    
    return loss

def response_format_reward(sample: dict, s: str) -> float:
    try:
        s = s.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[1]
    except:
        return -1
    if "<|eot_id|>" in s:
        s = s.split("<|eot_id|>")[0]
    total = 0; correct = 0
    tags = ["<thinking>", "</thinking>", "<answer>", "</answer>"]
    for tag in tags:
        if tag in s:
            total += 0.15
            if s.count(tag) > 1:
                total -= 0.01 * s.count(tag)
    if s.count("<thinking>") == 1: total += 0.5
    else: total -= 0.1
    if s.count("\n</thinking>\n<answer>\n") == 1: total += 1; correct += 1
    else:
        total += 0.2 if s.count("<thinking>") == 1 else -0.1
        total += 0.2 if s.count("<answer>") == 1 else -0.1
    if s.count("\n</answer>") == 1 and s.split("</answer>")[1].strip() == "": total += 1
    else: total -= 0.1
    if s.count("<answer>") == 1:
        total += 0.2
        r = s.split("<answer>")[1].split("</answer>")[0].strip()
        try:
            v = float(r); total += 1
            if v == float(sample.get("answer", 0)): total += 2; correct += 1
        except: total -= 0.1
    if correct == 3: total += 2
    return total + 0.001 * len(s)/512

def prepare_dataset(ds: Dataset, config: TrainingConfig) -> Dataset:
    def extract(ans: str) -> Optional[str]:
        return ans.split("####")[1].strip() if "####" in ans else None
    
    def proc(ex: dict) -> Optional[dict]:
        a = extract(ex["answer"])
        if a is None: return None
        return {
            "prompt": [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": ex["question"]}
            ],
            "answer": a
        }
    
    ds = ds.map(proc, remove_columns=[c for c in ds.column_names if c not in ["prompt", "answer"]])
    return ds.filter(lambda x: x is not None)

def sample_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    ds_iter: iter,
    config: TrainingConfig,
    fabric_instance: Optional[Fabric] = None
) -> tuple:
    with torch.no_grad():  # Ensure no gradients during sampling
        item = next(ds_iter)
        formatted = tokenizer.apply_chat_template(item["prompt"], tokenize=False)
        enc = tokenizer([formatted], padding=True, return_tensors="pt")
        input_ids = fabric_instance.to_device(enc.input_ids) if fabric_instance else enc.input_ids.to(config.device)
        prompt_len = input_ids.size(1)

        reps = input_ids.repeat(config.group_size, 1)
        
        # Clear memory before generation
        torch.cuda.empty_cache()
        
        # Beautiful generation progress
        with console.status("[bold green]🤖 Generating responses...", spinner="dots"):
            outputs = model.generate(
                reps, 
                max_new_tokens=config.max_new_tokens, 
                temperature=0.6,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True  # Enable KV cache for memory efficiency
            )
        
        # Clear intermediate variables to save memory
        del reps
        torch.cuda.empty_cache()
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        
        # Display generation results
        console.print("─" * 100, style="dim")
        dec = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Show the first generated sample in a beautiful panel
        sample_panel = Panel(
            dec[0][:500] + "..." if len(dec[0]) > 500 else dec[0],  # Truncate display to save memory
            title="📝 Generated Sample",
            box=box.ROUNDED,
            border_style="green"
        )
        console.print(sample_panel)
        
        avg_len = sum(len(d) for d in dec) / len(dec)
        console.print(f"[dim]Average response length: {avg_len:.1f} characters[/dim]")
        console.print("─" * 100, style="dim")
        
        rewards = torch.tensor(
            [response_format_reward(item, d) for d in decoded],
            dtype=config.dtype,
            device=outputs.device
        )

        mean_r = rewards.mean().repeat(config.group_size)
        std_r = rewards.std().repeat(config.group_size)

        mask = torch.zeros_like(outputs, dtype=torch.bool, device=outputs.device)
        gen = outputs[:, prompt_len:]
        mask[:, prompt_len:] = gen != tokenizer.pad_token_id

        # Clear more intermediate variables
        del decoded, dec, gen
        torch.cuda.empty_cache()

        return outputs, rewards, mask[:, 1:], mean_r, std_r, avg_len

def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in the checkpoint directory"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # First, try to find the latest_checkpoint.pt file
    latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    if os.path.exists(latest_checkpoint_path):
        return latest_checkpoint_path
    
    # If not found, look for numbered checkpoints
    checkpoint_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("checkpoint_") and filename.endswith(".pt"):
            try:
                iteration = int(filename.split("_")[1].split(".")[0])
                checkpoint_files.append((iteration, os.path.join(checkpoint_dir, filename)))
            except (ValueError, IndexError):
                continue
    
    if checkpoint_files:
        # Return the checkpoint with the highest iteration number
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)
        return checkpoint_files[0][1]
    
    return None

def validate_checkpoint_compatibility(checkpoint: dict, config: TrainingConfig, fabric: Fabric) -> bool:
    """Validate that a checkpoint is compatible with the current configuration"""
    try:
        if "config" not in checkpoint:
            fabric.print("⚠ Warning: Checkpoint doesn't contain config metadata")
            return True  # Allow loading for backward compatibility
        
        checkpoint_config = checkpoint["config"]
        
        # Check critical configuration parameters
        critical_params = ["model_name", "learning_rate", "group_size", "batch_size"]
        
        for param in critical_params:
            if param in checkpoint_config:
                checkpoint_value = checkpoint_config[param]
                current_value = getattr(config, param)
                
                if checkpoint_value != current_value:
                    fabric.print(f"⚠ Warning: Config mismatch for {param}")
                    fabric.print(f"  Checkpoint: {checkpoint_value}")
                    fabric.print(f"  Current:    {current_value}")
                    
                    # For some parameters, we can continue with a warning
                    if param in ["learning_rate"]:
                        fabric.print(f"  Continuing with current value: {current_value}")
                    elif param in ["model_name"]:
                        fabric.print("  ⚠ Model name mismatch - this may cause issues!")
        
        # Check if checkpoint has all required keys
        required_keys = ["model", "optimizer", "scheduler", "iteration"]
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            fabric.print(f"⚠ Warning: Checkpoint missing keys: {missing_keys}")
            return False
        
        return True
        
    except Exception as e:
        fabric.print(f"⚠ Warning: Error validating checkpoint: {str(e)}")
        return True  # Allow loading despite validation errors

def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer,
    scheduler,
    fabric: Fabric,
    config: TrainingConfig
) -> tuple[int, int]:
    """Load a checkpoint and return the iteration number and dataset position"""
    try:
        console.print(create_status_panel("📂 Loading Checkpoint", f"Loading from: {checkpoint_path}", "blue"))
        
        # Check if file exists first
        if not os.path.exists(checkpoint_path):
            console.print(create_status_panel("❌ File Not Found", f"Checkpoint file does not exist: {checkpoint_path}", "red"))
            return 0, 0
        
        # Check file size
        file_size = os.path.getsize(checkpoint_path)
        if file_size == 0:
            console.print(create_status_panel("❌ Empty File", "Checkpoint file is empty (0 bytes)", "red"))
            return 0, 0
        elif file_size < 1024:  # Less than 1KB is suspicious
            console.print(create_status_panel("⚠️ Suspicious File", f"Checkpoint file is very small ({file_size} bytes)", "yellow"))
        
        checkpoint = fabric.load(checkpoint_path)
        
        # Validate checkpoint structure
        if not isinstance(checkpoint, dict):
            console.print(create_status_panel("❌ Invalid Format", f"Checkpoint is not a dictionary, got {type(checkpoint)}", "red"))
            return 0, 0
        
        # Check for required keys
        required_keys = ["model", "optimizer", "scheduler", "iteration"]
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            console.print(create_status_panel("❌ Missing Keys", f"Checkpoint is missing required keys: {missing_keys}", "red"))
            console.print(create_status_panel("💡 Available Keys", f"Found keys: {list(checkpoint.keys())}", "blue"))
            return 0, 0
        
        # Validate checkpoint compatibility
        if not validate_checkpoint_compatibility(checkpoint, config, fabric):
            console.print(create_status_panel("⚠️ Warning", "Checkpoint validation failed. Continuing with fresh training...", "yellow"))
            return 0, 0
        
        # Create loading progress table
        loading_table = Table(title="🔄 Checkpoint Loading Progress", box=box.SIMPLE)
        loading_table.add_column("Component", style="cyan")
        loading_table.add_column("Status", style="green")
        loading_table.add_column("Details", style="dim white")
        
        # Load model state
        try:
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
                loading_table.add_row("Model State", "✅ Loaded", "Neural network weights restored")
            else:
                loading_table.add_row("Model State", "⚠️ Missing", "No model state found in checkpoint")
        except Exception as e:
            loading_table.add_row("Model State", "❌ Failed", f"Error: {str(e)}")
            console.print(loading_table)
            console.print(create_status_panel("❌ Model Load Error", f"Failed to load model state: {str(e)}", "red"))
            return 0, 0
        
        # Load optimizer state
        try:
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
                loading_table.add_row("Optimizer State", "✅ Loaded", "Adam/AdamW state restored")
            else:
                loading_table.add_row("Optimizer State", "⚠️ Missing", "No optimizer state found in checkpoint")
        except Exception as e:
            loading_table.add_row("Optimizer State", "❌ Failed", f"Error: {str(e)}")
            console.print(loading_table)
            console.print(create_status_panel("❌ Optimizer Load Error", f"Failed to load optimizer state: {str(e)}", "red"))
            return 0, 0
        
        # Load scheduler state
        try:
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
                loading_table.add_row("Scheduler State", "✅ Loaded", "Learning rate schedule restored")
            else:
                loading_table.add_row("Scheduler State", "⚠️ Missing", "No scheduler state found in checkpoint")
        except Exception as e:
            loading_table.add_row("Scheduler State", "❌ Failed", f"Error: {str(e)}")
            console.print(loading_table)
            console.print(create_status_panel("❌ Scheduler Load Error", f"Failed to load scheduler state: {str(e)}", "red"))
            return 0, 0
        
        # Get iteration number and dataset position
        iteration = checkpoint.get("iteration", 0)
        dataset_position = checkpoint.get("dataset_position", iteration)  # Fallback to iteration for backward compatibility
        
        loading_table.add_row("Iteration", f"✅ {iteration}", "Training iteration number")
        loading_table.add_row("Dataset Position", f"✅ {dataset_position}", "Samples processed from dataset")
        
        console.print(loading_table)
        
        # Check for iteration 0 specifically
        if iteration == 0:
            console.print(create_status_panel("⚠️ Iteration Zero", "Checkpoint was saved at iteration 0 (beginning of training)", "yellow"))
        
        console.print(create_status_panel("✅ Success", f"Checkpoint loaded! Resuming from iteration {iteration}, dataset position {dataset_position}", "green"))
        
        return iteration, dataset_position
        
    except Exception as e:
        error_msg = str(e)
        console.print(create_status_panel("❌ Error", f"Failed to load checkpoint: {error_msg}\nContinuing with fresh training...", "red"))
        
        # Provide specific debugging advice
        if "pickle" in error_msg.lower() or "unpickling" in error_msg.lower():
            console.print(create_status_panel("💡 Debug Tip", "Checkpoint file appears corrupted. Try:\n1. Delete the checkpoint file\n2. Restart training\n3. Check disk space", "blue"))
        elif "fabric" in error_msg.lower():
            console.print(create_status_panel("💡 Debug Tip", "Fabric loading issue. The checkpoint might be from a different Fabric version.", "blue"))
        elif "state_dict" in error_msg.lower():
            console.print(create_status_panel("💡 Debug Tip", "Model state dictionary mismatch. The checkpoint might be from a different model configuration.", "blue"))
        
        return 0, 0

def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        return allocated, reserved
    return 0, 0

def initialize_training(config: TrainingConfig, fabric: Optional[Fabric] = None):
    """Initialize the model, tokenizer, optimizer, and other components for training"""
    
    # Display beautiful configuration
    display_config(config)
    
    if fabric is None:
        logger = WandbLogger(project=config.project_name) if config.log_wandb else None
        fabric = Fabric(
            accelerator="cuda",
            devices=1,
            precision="bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed",
            loggers=logger
        )
        fabric.launch()
    
    fabric.seed_everything(42)
    
    console.print(create_status_panel("🤖 Loading Model", f"Loading {config.model_name}...", "blue"))
    
    # Memory monitoring before model loading
    alloc_before, reserved_before = get_memory_usage()
    console.print(f"[dim]GPU Memory before model loading: {alloc_before:.2f}GB allocated, {reserved_before:.2f}GB reserved[/dim]")
    
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with memory-efficient settings
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=config.dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        use_cache=False  # Disable cache during training to save memory
    )
    
    # Memory monitoring after model loading
    alloc_after, reserved_after = get_memory_usage()
    console.print(f"[dim]GPU Memory after model loading: {alloc_after:.2f}GB allocated, {reserved_after:.2f}GB reserved[/dim]")
    console.print(f"[dim]Model loading used: {alloc_after - alloc_before:.2f}GB[/dim]")
    
    console.print(create_status_panel("🔧 Configuring LoRA", f"Rank: {config.lora_r}, Alpha: {config.lora_alpha}", "blue"))
    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        use_dora=True,
    )
    model = get_peft_model(model, lora_cfg)

    console.print(create_status_panel("⚙️ Setting up Optimizer", f"AdamW with LR: {config.learning_rate:.2e}", "blue"))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.optimizer_betas
    )
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=config.max_iterations)
    
    model, optimizer = fabric.setup(model, optimizer)
    model.mark_forward_method('generate')
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    
    ref_model = model
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
        
    # Prepare dataset
    console.print(create_status_panel("📚 Loading Dataset", "GSM8K dataset for math reasoning", "blue"))
    ds = load_dataset("openai/gsm8k", "main")["train"]
    ds = prepare_dataset(ds.shuffle(seed=42), config)
    ds_iter = iter(ds)
    
    start_iter = 0
    dataset_position = 0
    
    # Handle checkpoint resuming - ALWAYS enabled
    checkpoint_path = None

    # Option 1: Resume from a specific checkpoint (if provided)
    if config.resume_from:
        if os.path.exists(config.resume_from):
            checkpoint_path = config.resume_from
            console.print(create_status_panel("🔄 Resuming", f"From specified checkpoint: {config.resume_from}", "yellow"))
        else:
            console.print(create_status_panel("⚠️ Warning", f"Specified checkpoint not found: {config.resume_from}\nLooking for latest checkpoint instead...", "yellow"))

    # Option 2: Always look for the latest checkpoint (default behavior)
    if checkpoint_path is None:  # Only if no specific checkpoint was found
        checkpoint_path = find_latest_checkpoint(config.checkpoint_dir)
        if checkpoint_path:
            console.print(create_status_panel("🔄 Auto-Resume", f"Latest checkpoint found: {checkpoint_path}", "yellow"))
        else:
            console.print(create_status_panel("🆕 Fresh Start", "No existing checkpoint found. Starting fresh training...", "green"))
    
    # Load the checkpoint if one was found
    if checkpoint_path:
        start_iter, dataset_position = load_checkpoint(checkpoint_path, model, optimizer, scheduler, fabric, config)
        
        # Advance the dataset iterator to the correct position
        if dataset_position > 0:
            console.print(create_status_panel("📖 Dataset Position", f"Advancing dataset iterator by {dataset_position} samples...", "blue"))
            for _ in range(dataset_position):
                try:
                    next(ds_iter)
                except StopIteration:
                    # If we've exhausted the dataset, create a new iterator
                    # This handles cases where we've trained for multiple epochs
                    console.print(create_status_panel("🔄 Dataset Reset", "Dataset exhausted, creating new iterator...", "yellow"))
                    ds_iter = iter(ds)
                    break
        
        # Validate that the checkpoint was loaded properly
        if start_iter > 0:
            console.print(create_status_panel("✅ Resume Success", f"Loaded checkpoint from completed iteration {start_iter}", "green"))
            console.print(create_status_panel("📊 Resume Info", f"Next iteration to process: {start_iter + 1} → {config.max_iterations}", "blue"))
        else:
            console.print(create_status_panel("⚠️ Warning", "Checkpoint loading failed or iteration was 0", "yellow"))
            dataset_position = 0
    else:
        console.print(create_status_panel("🚀 Ready to Train", "Starting fresh training from iteration 0", "green"))
    
    # Display model info
    model_info = Table(title="🧠 Model Information", box=box.ROUNDED)
    model_info.add_column("Attribute", style="cyan")
    model_info.add_column("Value", style="green")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_info.add_row("Total Parameters", f"{total_params:,}")
    model_info.add_row("Trainable Parameters", f"{trainable_params:,}")
    model_info.add_row("Trainable Percentage", f"{100 * trainable_params / total_params:.2f}%")
    model_info.add_row("Model Type", model.__class__.__name__)
    model_info.add_row("Device", str(next(model.parameters()).device))
    
    console.print(model_info)
    console.print()
    
    return model, ref_model, tokenizer, optimizer, scheduler, ds_iter, start_iter, fabric

def save_checkpoint(model, optimizer, scheduler, iteration, config, fabric):
    """Save a checkpoint of the training state"""
    try:
        # Create checkpoint directory if it doesn't exist
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_{iteration}.pt")
        latest_checkpoint_path = os.path.join(config.checkpoint_dir, "latest_checkpoint.pt")
        temp_checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_{iteration}_temp.pt")
        temp_latest_path = os.path.join(config.checkpoint_dir, "latest_checkpoint_temp.pt")
        
        # Create checkpoint with metadata
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "iteration": iteration,  # The iteration number we just completed
            "dataset_position": iteration,  # Track how many samples we've consumed
            "config": {
                "model_name": config.model_name,
                "learning_rate": config.learning_rate,
                "group_size": config.group_size,
                "batch_size": config.batch_size,
                "max_iterations": config.max_iterations,
            },
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        # Show saving progress with clear messaging
        console.print(create_status_panel("💾 Saving Checkpoint", 
            f"Completed iteration {iteration} → Saving to {checkpoint_path}", "blue"))
        
        # Save to temporary files first to ensure atomic writes
        fabric.save(temp_checkpoint_path, checkpoint)
        fabric.save(temp_latest_path, checkpoint)
        
        # Move temporary files to final locations (atomic operation)
        if os.path.exists(temp_checkpoint_path):
            os.rename(temp_checkpoint_path, checkpoint_path)
        if os.path.exists(temp_latest_path):
            os.rename(temp_latest_path, latest_checkpoint_path)
        
        console.print(create_status_panel("✅ Checkpoint Saved", 
            f"State saved after completing iteration {iteration} • Next iteration will be {iteration + 1}", "green"))
        
        # Clean up old checkpoints (keep only the last checkpoint to save disk space)
        cleanup_old_checkpoints(config.checkpoint_dir, keep_last=1)
        
    except Exception as e:
        console.print(create_status_panel("❌ Save Error", f"Failed to save checkpoint: {str(e)}", "red"))
        # Clean up temporary files if they exist
        for temp_path in [temp_checkpoint_path, temp_latest_path]:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

def cleanup_old_checkpoints(checkpoint_dir: str, keep_last: int = 1):
    """Clean up old checkpoints, keeping only the most recent one to save disk space"""
    try:
        checkpoint_files = []
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith("checkpoint_") and filename.endswith(".pt") and "_temp" not in filename:
                try:
                    iteration = int(filename.split("_")[1].split(".")[0])
                    checkpoint_files.append((iteration, os.path.join(checkpoint_dir, filename)))
                except (ValueError, IndexError):
                    continue
        
        # Sort by iteration number and remove old checkpoints
        if len(checkpoint_files) > keep_last:
            checkpoint_files.sort(key=lambda x: x[0], reverse=True)
            removed_count = 0
            for _, checkpoint_path in checkpoint_files[keep_last:]:
                try:
                    filename = os.path.basename(checkpoint_path)
                    os.remove(checkpoint_path)
                    removed_count += 1
                except:
                    pass
            
            if removed_count > 0:
                console.print(create_status_panel("🧹 Cleanup", 
                    f"Removed {removed_count} old checkpoint{'s' if removed_count > 1 else ''} to save disk space", "blue"))
    except Exception:
        # Silently ignore cleanup errors
        pass

def train(config: TrainingConfig, fabric: Optional[Fabric] = None) -> None:
    model, ref_model, tokenizer, optimizer, scheduler, ds_iter, iters, fabric = initialize_training(config, fabric)

    hyper_params = {
        "model_name": config.model_name,
        "group_size": config.group_size,
        "batch_size": config.batch_size,
        "micro_batch_size": config.micro_batch_size,
        "max_iterations": config.max_iterations,
        "max_new_tokens": config.max_new_tokens,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "beta": config.beta,
        "epsilon_low": config.epsilon_low,
        "epsilon_high": config.epsilon_high,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "target_modules": config.target_modules,
        "optimizer_betas": config.optimizer_betas,
    }
    if fabric.logger:
        fabric.logger.log_hyperparams(hyper_params)
    
    start = time.perf_counter()
    
    # Beautiful training start message
    if iters > 0:
        console.print(create_status_panel("🔄 Resuming Training", f"Continuing from iteration {iters + 1} to {config.max_iterations}", "yellow"))
    else:
        console.print(create_status_panel("🚀 Training Started", f"Beginning training for {config.max_iterations} iterations", "green"))
    
    with fabric.rank_zero_first():
        pass  # Remove the old fabric.print

    while iters < config.max_iterations:
        # Calculate which iteration we're about to process
        current_iter = iters + 1
        
        console.print(f"[dim]Processing iteration {current_iter}...[/dim]")
        
        # Aggressive memory cleanup before each iteration
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        with torch.no_grad():
            # Process old logps in smaller chunks to save memory
            outputs, rewards, mask, mean_r, std_r, avg_len = sample_batch(
                model, tokenizer, ds_iter, config, fabric
            )
            
            # Move outputs to CPU immediately to free GPU memory
            outputs_cpu = outputs.cpu()
            del outputs
            torch.cuda.empty_cache()
            
            # Process old_logps in micro-batches and keep on CPU
            old_logps = []
            for idx in range(0, outputs_cpu.size(0), config.micro_batch_size):
                sample = outputs_cpu[idx:idx + config.micro_batch_size].cuda()
                batch_logps = get_per_token_logps(model, sample).cpu()
                old_logps.append(batch_logps)
                del sample, batch_logps
                torch.cuda.empty_cache()
            
            old_logps = torch.cat(old_logps, dim=0)

        # Process training in micro-batches with aggressive memory management
        total_loss = 0.0
        num_micro_batches = 0
        
        for idx in range(0, outputs_cpu.size(0), config.micro_batch_size):
            micro_outputs = outputs_cpu[idx:idx + config.micro_batch_size].cuda()
            micro_old_logps = old_logps[idx:idx + config.micro_batch_size].cuda()
            micro_rewards = rewards[idx:idx + config.micro_batch_size]
            micro_mean_r = mean_r[idx:idx + config.micro_batch_size]
            micro_std_r = std_r[idx:idx + config.micro_batch_size]
            micro_mask = mask[idx:idx + config.micro_batch_size]
            
            loss = compute_loss(
                model,
                ref_model,
                micro_outputs,
                micro_old_logps,
                micro_rewards,
                micro_mean_r,
                micro_std_r,
                micro_mask,
                config,
                fabric
            )
            
            # Scale loss by gradient accumulation steps
            loss = loss / config.gradient_accumulation_steps
            
            # Accumulate loss for logging
            total_loss += loss.item()
            num_micro_batches += 1
            
            fabric.backward(loss)
            
            # Clean up micro-batch variables immediately
            del micro_outputs, micro_old_logps, micro_rewards, micro_mean_r, micro_std_r, micro_mask, loss
            torch.cuda.empty_cache()
        
        # Calculate average loss across micro-batches (scale back up for logging)
        avg_loss = (total_loss * config.gradient_accumulation_steps) / num_micro_batches if num_micro_batches > 0 else 0.0
        
        # Only step optimizer every gradient_accumulation_steps iterations
        if (iters + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # NOW we have completed the iteration - increment the counter
        iters += 1  # iters now equals the iteration number we just completed
        
        # Log metrics every iteration for better monitoring
        metrics = {
            "iter": iters,
            "loss": avg_loss,
            "reward": rewards.mean().item(),
            "avg_len": avg_len,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        
        # Beautiful metrics display
        metrics_table = Table(title=f"📊 Training Metrics - Iteration {iters}", box=box.SIMPLE)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")
        metrics_table.add_column("Status", style="yellow")
        
        # Progress percentage
        progress_pct = (iters / config.max_iterations) * 100
        
        metrics_table.add_row("Loss", f"{avg_loss:.4f}", "🔥" if avg_loss < 1.0 else "📈")
        metrics_table.add_row("Avg Reward", f"{rewards.mean().item():.4f}", "🎯" if rewards.mean().item() > 0 else "📉")
        metrics_table.add_row("Learning Rate", f"{scheduler.get_last_lr()[0]:.2e}", "⚡")
        metrics_table.add_row("Progress", f"{progress_pct:.1f}%", "🚀")
        metrics_table.add_row("Time Elapsed", str(datetime.timedelta(seconds=int(time.perf_counter()-start))), "⏱️")
        
        # Memory usage
        alloc_gb, reserved_gb = get_memory_usage()
        metrics_table.add_row("GPU Memory", f"Alloc: {alloc_gb:.2f}GB | Reserved: {reserved_gb:.2f}GB", "💾")
        
        console.print(metrics_table)
        console.print()
        
        with fabric.rank_zero_first():
            if fabric.logger:
                fabric.log_dict(metrics, step=iters)
        
        # Save checkpoint at specified intervals - we save the state after completing iteration 'iters'
        if iters % config.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, iters, config, fabric)
        
        del outputs_cpu, rewards, mask, mean_r, std_r
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    console.print(create_status_panel("🎉 Training Complete", f"Successfully completed {config.max_iterations} iterations!", "green"))
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, iters, config, fabric)
    
    # Save final model in HuggingFace format
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    model_path = f"model-nanoGRPO-{timestamp}"
    
    console.print(create_status_panel("💾 Saving Final Model", f"Saving to {model_path}", "blue"))
    
    with fabric.rank_zero_first():
        fabric.save(f"{model_path}/model.pt", {"model": model.state_dict()})
        model.save_pretrained(f"{model_path}/hf_model")
        tokenizer.save_pretrained(f"{model_path}/hf_tokenizer")
        
    console.print(create_status_panel("✅ Model Saved", f"Final model saved to {model_path}", "green"))

def display_config(config: TrainingConfig):
    """Display training configuration in a beautiful table"""
    
    # Create main configuration table
    config_table = Table(title="🚀 Training Configuration", box=box.ROUNDED, title_style="bold magenta")
    config_table.add_column("Parameter", style="cyan", no_wrap=True)
    config_table.add_column("Value", style="green")
    config_table.add_column("Description", style="dim white")
    
    # Model configuration
    config_table.add_row("Model", config.model_name, "Base model for training")
    config_table.add_row("Tokenizer", config.tokenizer_name, "Tokenizer for text processing")
    
    # Training hyperparameters
    config_table.add_row("Group Size", str(config.group_size), "Number of samples per group")
    config_table.add_row("Batch Size", str(config.batch_size), "Training batch size")
    config_table.add_row("Micro Batch Size", str(config.micro_batch_size), "Micro batch size for memory optimization")
    config_table.add_row("Max Iterations", str(config.max_iterations), "Maximum training iterations")
    config_table.add_row("Max New Tokens", str(config.max_new_tokens), "Maximum tokens to generate")
    config_table.add_row("Learning Rate", f"{config.learning_rate:.2e}", "Optimizer learning rate")
    config_table.add_row("Weight Decay", str(config.weight_decay), "L2 regularization strength")
    config_table.add_row("Beta", str(config.beta), "KL divergence regularization")
    config_table.add_row("Epsilon Low", str(config.epsilon_low), "PPO clipping lower bound")
    config_table.add_row("Epsilon High", str(config.epsilon_high), "PPO clipping upper bound")
    
    # LoRA configuration
    config_table.add_row("LoRA Rank", str(config.lora_r), "LoRA adaptation rank")
    config_table.add_row("LoRA Alpha", str(config.lora_alpha), "LoRA scaling parameter")
    
    # Checkpointing
    config_table.add_row("Checkpoint Dir", config.checkpoint_dir, "Directory for saving checkpoints")
    config_table.add_row("Save Every", f"{config.save_every} iterations", "Checkpoint save frequency")
    config_table.add_row("Auto Resume", "✅ Always Enabled", "Automatic checkpoint resuming")
    config_table.add_row("Wandb Logging", "✅" if config.log_wandb else "❌", "Weights & Biases logging")
    
    console.print(config_table)
    console.print()

def create_status_panel(title: str, message: str, style: str = "white") -> Panel:
    """Create a beautiful status panel"""
    return Panel(
        Align.center(Text(message, style=style)),
        title=title,
        box=box.ROUNDED,
        border_style=style
    )

def inspect_checkpoint(checkpoint_path: str):
    """Inspect a checkpoint file and display its contents for debugging"""
    if not os.path.exists(checkpoint_path):
        console.print(create_status_panel("❌ File Not Found", f"Checkpoint file does not exist: {checkpoint_path}", "red"))
        return
    
    try:
        # Get file info
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        file_time = datetime.datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
        
        console.print(create_status_panel("🔍 Inspecting Checkpoint", f"File: {checkpoint_path}", "blue"))
        
        # Try to load with torch directly (without fabric)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create inspection table
        inspect_table = Table(title="📋 Checkpoint Contents", box=box.ROUNDED)
        inspect_table.add_column("Property", style="cyan")
        inspect_table.add_column("Value", style="green")
        inspect_table.add_column("Details", style="dim white")
        
        # File information
        inspect_table.add_row("File Size", f"{file_size:.2f} MB", "Physical file size")
        inspect_table.add_row("Modified", file_time.strftime("%Y-%m-%d %H:%M:%S"), "Last modification time")
        
        # Checkpoint structure
        if isinstance(checkpoint, dict):
            inspect_table.add_row("Type", "Dictionary", "Standard checkpoint format")
            
            # Check required keys
            required_keys = ["model", "optimizer", "scheduler", "iteration"]
            missing_keys = [key for key in required_keys if key not in checkpoint]
            
            if missing_keys:
                inspect_table.add_row("Missing Keys", str(missing_keys), "❌ Required keys not found")
            else:
                inspect_table.add_row("Required Keys", "✅ All present", "model, optimizer, scheduler, iteration")
            
            # Show available keys
            all_keys = list(checkpoint.keys())
            inspect_table.add_row("Available Keys", str(all_keys), "All keys in checkpoint")
            
            # Show iteration info
            iteration = checkpoint.get("iteration", "NOT_FOUND")
            dataset_position = checkpoint.get("dataset_position", "NOT_FOUND")
            inspect_table.add_row("Iteration", str(iteration), "Training iteration number")
            inspect_table.add_row("Dataset Position", str(dataset_position), "Dataset samples processed")
            
            # Show config if available
            if "config" in checkpoint:
                config_info = checkpoint["config"]
                inspect_table.add_row("Config Keys", str(list(config_info.keys())), "Saved configuration")
                if "model_name" in config_info:
                    inspect_table.add_row("Model Name", config_info["model_name"], "Model used in checkpoint")
            else:
                inspect_table.add_row("Config", "❌ Missing", "No configuration metadata")
            
            # Show timestamp if available
            if "timestamp" in checkpoint:
                inspect_table.add_row("Timestamp", checkpoint["timestamp"], "When checkpoint was created")
        else:
            inspect_table.add_row("Type", type(checkpoint).__name__, "❌ Unexpected format")
        
        console.print(inspect_table)
        
        # Validation check
        if isinstance(checkpoint, dict) and not missing_keys and iteration != "NOT_FOUND":
            if iteration == 0:
                console.print(create_status_panel("⚠️ Warning", "Iteration is 0 - this checkpoint was saved at the beginning of training", "yellow"))
            else:
                console.print(create_status_panel("✅ Looks Good", f"Checkpoint appears valid with iteration {iteration}", "green"))
        else:
            console.print(create_status_panel("❌ Issues Found", "Checkpoint has structural problems", "red"))
        
    except Exception as e:
        console.print(create_status_panel("❌ Load Error", f"Failed to load checkpoint: {str(e)}", "red"))
        
        # Try to give specific advice based on error type
        error_str = str(e).lower()
        if "pickle" in error_str:
            console.print(create_status_panel("💡 Suggestion", "Checkpoint file may be corrupted. Try deleting it and starting fresh.", "yellow"))
        elif "permission" in error_str:
            console.print(create_status_panel("💡 Suggestion", "Permission issue. Check file permissions.", "yellow"))
        elif "disk" in error_str or "space" in error_str:
            console.print(create_status_panel("💡 Suggestion", "Disk space issue. Check available storage.", "yellow"))
        else:
            console.print(create_status_panel("💡 Suggestion", "Try deleting the checkpoint file and starting fresh training.", "yellow"))

def display_startup_banner():
    """Display a beautiful startup banner"""
    banner_text = """
    ███╗   ██╗ █████╗ ███╗   ██╗ ██████╗      ██████╗ ██████╗ ██████╗  ██████╗ 
    ████╗  ██║██╔══██╗████╗  ██║██╔═══██╗    ██╔════╝ ██╔══██╗██╔══██╗██╔═══██╗
    ██╔██╗ ██║███████║██╔██╗ ██║██║   ██║    ██║  ███╗██████╔╝██████╔╝██║   ██║
    ██║╚██╗██║██╔══██║██║╚██╗██║██║   ██║    ██║   ██║██╔══██╗██╔═══╝ ██║   ██║
    ██║ ╚████║██║  ██║██║ ╚████║╚██████╔╝    ╚██████╔╝██║  ██║██║     ╚██████╔╝
    ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝      ╚═════╝ ╚═╝  ╚═╝╚═╝      ╚═════╝ 
    """
    
    console.print(Panel(
        Align.center(Text(banner_text, style="bold magenta")),
        title="🚀 Generalized Reward Policy Optimization",
        subtitle="✨ Beautiful Terminal Training Experience ✨",
        box=box.DOUBLE,
        border_style="magenta"
    ))
    console.print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train GRPO model with automatic checkpoint resuming")
    parser.add_argument("--resume-from", type=str, help="Path to specific checkpoint to resume from (overrides auto-detection)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to store checkpoints")
    parser.add_argument("--max-iterations", type=int, help="Maximum number of training iterations")
    parser.add_argument("--learning-rate", type=float, help="Learning rate for training")
    parser.add_argument("--group-size", type=int, help="Group size for training")
    parser.add_argument("--batch-size", type=int, help="Batch size for training")
    parser.add_argument("--save-every", type=int, help="Save checkpoint every N iterations")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--inspect-checkpoint", type=str, help="Inspect a checkpoint file for debugging (don't start training)")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode with fewer iterations for checkpoint testing")
    
    args = parser.parse_args()
    
    # Display beautiful startup banner first
    display_startup_banner()
    
    # Handle checkpoint inspection mode
    if args.inspect_checkpoint:
        console.print(create_status_panel("🔍 Checkpoint Inspector", "Analyzing checkpoint file...", "blue"))
        inspect_checkpoint(args.inspect_checkpoint)
        exit(0)
    
    config = TrainingConfig()
    
    # Override config with command line arguments FIRST
    if args.resume_from:
        config.resume_from = args.resume_from
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.max_iterations:
        config.max_iterations = args.max_iterations
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.group_size:
        config.group_size = args.group_size
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.save_every:
        config.save_every = args.save_every
    if args.no_wandb:
        config.log_wandb = False
    
    # Test mode configuration (only override if not explicitly set)
    if args.test_mode:
        if not args.max_iterations:  # Only set if not specified via command line
            config.max_iterations = 8
        if not args.save_every:  # Only set if not specified via command line
            config.save_every = 4
        if not args.group_size:  # Only set if not specified via command line
            config.group_size = 2
        if not args.batch_size:  # Only set if not specified via command line
            config.batch_size = 2
        config.log_wandb = False
        console.print(create_status_panel("🧪 Test Mode", "Running with reduced parameters for checkpoint testing", "cyan"))
    
    # Beautiful configuration summary
    console.print()
    console.rule("[bold magenta]🎯 Final Training Configuration", style="magenta")
    
    config_summary = Table(box=box.SIMPLE, show_header=False)
    config_summary.add_column("Setting", style="cyan", no_wrap=True)
    config_summary.add_column("Value", style="green")
    
    config_summary.add_row("Auto-resume", "✅ Always Enabled")
    config_summary.add_row("Resume from", config.resume_from or "None")
    config_summary.add_row("Checkpoint dir", config.checkpoint_dir)
    config_summary.add_row("Max iterations", str(config.max_iterations))
    config_summary.add_row("Learning rate", f"{config.learning_rate:.2e}")
    config_summary.add_row("Save every", f"{config.save_every} iterations")
    config_summary.add_row("Wandb logging", "✅ Enabled" if config.log_wandb else "❌ Disabled")
    
    console.print(config_summary)
    console.print()
    
    strategy = DeepSpeedStrategy(cpu_checkpointing=True)
    logger = WandbLogger(project=config.project_name) if config.log_wandb else None
    
    tensorboard_logger = TensorBoardLogger(save_dir="logs")
    fabric_instance = Fabric(
        accelerator="cuda",
        devices=1,
        precision="bf16-true",
        loggers=[logger],
        # strategy=strategy
    )
    fabric_instance.launch()
    train(config, fabric_instance)
