from fuchsia.vllm_server import DataSamplerServer, ServerConfig, Rollout
from datasets import load_dataset
from rich import print
from typing import Optional, List
from datasets import Dataset
from transformers import AutoTokenizer
from pathlib import Path    

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SYSTEM_PROMPT = "Respond in following format:<thinking>{think on problem, understand problem create plan to solve it relfect on your step answer only when you are sure}</thinking><answer>{number}</answer>"


def response_format_reward(sample: dict, s: str, *args, **kwargs) -> float:
    """Improved reward function with better validation and scoring."""
    START_OF_TEXT_TOKEN = "<|im_start|>"
    END_OF_TEXT_TOKEN = "<|eot_id|>"
    START_HEADER_TOKEN = "<|start_header_id|>"
    END_HEADER_TOKEN = "<|end_header_id|>"
    ASSISTANT_TOKEN = "assistant"
    USER_TOKEN = "user"

    START_THINKING_TOKEN = "<thinking>"
    END_THINKING_TOKEN = "\n</thinking>"
    START_ANSWER_TOKEN = "\n<answer>"
    END_ANSWER_TOKEN = "\n</answer>"
    idx = kwargs["idx"]
    
    try:
        # Extract the actual response
        try:
            s = s.split(f"{START_HEADER_TOKEN}{ASSISTANT_TOKEN}{END_HEADER_TOKEN}")[1]
        except IndexError:
            return -1.0

        if END_OF_TEXT_TOKEN in s:
            s = s.split(END_OF_TEXT_TOKEN)[0]

        # Initialize reward components
        format_reward = 0.0
        content_reward = 0.0
        correct_template = 0

        # Check format tags
        required_tags = [
            START_THINKING_TOKEN,
            END_THINKING_TOKEN,
            START_ANSWER_TOKEN,
            END_ANSWER_TOKEN,
        ]
        for tag in required_tags:
            if tag in s:
                format_reward += 0.15
                if s.count(tag) > 1:
                    format_reward -= s.count(tag) * 0.01

        # Validate thinking section
        if s.count("<thinking>") == 1:
            format_reward += 0.5
            thinking_content = (
                s.split(START_THINKING_TOKEN)[1].split(END_THINKING_TOKEN)[0].strip()
            )
            if len(thinking_content) > 10:  # Basic content validation
                content_reward += 0.5
        else:
            format_reward -= 0.1
        
            
        # Validate answer section
        if "<answer>" in s and "</answer>" in s:
            format_reward += 0.4
            answer_content = (
                s.split(START_ANSWER_TOKEN)[1].split(END_ANSWER_TOKEN)[0].strip()
            )
            try:
                answer_value = float(answer_content)
                correct_template += 1
                if answer_value == float(sample["correct_answer"]):
                    content_reward += 4
            except ValueError:
                content_reward -= 0.1

        if correct_template == 1:
            format_reward += 1.0
        return format_reward + content_reward

    except Exception as e:
        print(f"[yellow]Error in reward calculation: {e}[/yellow]")
        return -1.0


idx = 0
def reward_function_1(rollouts: List[Rollout], *args, **kwargs):
    global idx
    idx += 1
    print(f"{idx=}")
    lst = []
    for rollout in rollouts:
        reward = response_format_reward(rollout.item, rollout.completion , idx=idx)
        if idx > 8*8 and reward < 3:
            reward = 0
        lst.append(reward)
        
    if idx > 8*8 and not(any(x>4 for x in lst)):
        lst = [0 for _ in lst]
        
    return lst


def prepare_dataset(dataset, tokenizer) -> Dataset:
    """Prepare the GSM8K dataset with better error handling and validation."""

    def extract_hash_answer(text: str) -> Optional[str]:
        try:
            if "####" not in text:
                return None
            answer = text.split("####")[1].strip()
            answer = answer.replace(",", "").strip()
            answer = float(answer)
            return answer
        except (ValueError, IndexError):
            return None

    def process_example(example: dict) -> Optional[dict]:
        # print(f"{example=}")
        try:
            answer = extract_hash_answer(example["answer"])
            example["correct_answer"] = answer
            if answer is None:
                print("FUCKKKKKKKKK.")
                print(f"{example['answer']=}")
                return None
            example["text"] = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["question"]},
                ],
                tokenize=False,
            )
            return example
        except Exception as e:
            print(f"[yellow]Failed to process example: {e}[/yellow]")
            return None

    try:
        dataset = dataset.map(
            process_example,
            desc="Processing dataset",
        )
        dataset = dataset.filter(lambda x: x is not None)
        print(f"[green]Processed dataset size: {len(dataset)}[/green]")
        return dataset
    except Exception as e:
        print(f"[red]Failed to prepare dataset: {e}[/red]")
        raise



def main():


    server_config = ServerConfig.from_yaml(Path(__file__).parent / "gsm8k_config.yaml")
    tokenizer = AutoTokenizer.from_pretrained(server_config.model)
    dataset = load_dataset(server_config.dataset_name, server_config.dataset_split)["train"]
    dataset = dataset.shuffle()
    dataset = prepare_dataset(dataset, tokenizer)
    
    server = DataSamplerServer(server_config, dataset, [reward_function_1])
    server.serve()


if __name__ == "__main__":
    main()
    