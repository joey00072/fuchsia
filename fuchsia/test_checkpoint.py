#!/usr/bin/env python3
"""
Simple test script to verify checkpoint saving and loading functionality.
This script will:
1. Run training for a few iterations
2. Save a checkpoint at iteration 4
3. Stop and restart from that checkpoint
4. Verify it continues from iteration 5 (not 4 again)
"""

import os
import shutil
from interleave_grpo import TrainingConfig, train, find_latest_checkpoint, inspect_checkpoint
from lightning.fabric import Fabric
from lightning.pytorch.loggers import TensorBoardLogger

def test_checkpoint_functionality():
    """Test checkpoint save and resume functionality"""
    
    # Clean up any existing checkpoints
    checkpoint_dir = "test_checkpoints"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    
    print("🧪 Testing Checkpoint Functionality")
    print("=" * 50)
    
    # Create test configuration
    config = TrainingConfig(
        max_iterations=8,  # Run for 8 iterations total
        save_every=4,      # Save every 4 iterations
        checkpoint_dir=checkpoint_dir,
        auto_resume=False,  # Disable auto-resume for controlled testing
        log_wandb=False,    # Disable wandb for testing
        group_size=2,       # Smaller group size for faster testing
        batch_size=2,       # Smaller batch size
    )
    
    print(f"📁 Using checkpoint directory: {checkpoint_dir}")
    print(f"💾 Will save checkpoints every {config.save_every} iterations")
    print(f"🎯 Target: {config.max_iterations} total iterations")
    print()
    
    # Test Phase 1: Initial training (should save at iteration 4)
    print("🚀 Phase 1: Initial Training (0 → 4 iterations)")
    print("-" * 30)
    
    # Limit to 4 iterations for first run
    config.max_iterations = 4
    
    # Setup fabric for testing
    tensorboard_logger = TensorBoardLogger(save_dir="test_logs")
    fabric = Fabric(
        accelerator="cuda",
        devices=1,
        precision="bf16-true",
        loggers=[tensorboard_logger],
    )
    fabric.launch()
    
    # Run initial training
    train(config, fabric)
    
    # Check that checkpoint was created
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_4.pt")
    if os.path.exists(checkpoint_path):
        print(f"✅ Checkpoint saved successfully: {checkpoint_path}")
        inspect_checkpoint(checkpoint_path)
    else:
        print(f"❌ Checkpoint NOT found: {checkpoint_path}")
        return False
    
    print()
    
    # Test Phase 2: Resume training (should continue from iteration 5)
    print("🔄 Phase 2: Resume Training (4 → 8 iterations)")
    print("-" * 30)
    
    # Reset max_iterations for full run
    config.max_iterations = 8
    config.resume_from = checkpoint_path  # Explicitly resume from checkpoint_4.pt
    
    print(f"📂 Resuming from: {config.resume_from}")
    
    # Run resumed training
    train(config, fabric)
    
    # Check that final checkpoint was created
    final_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_8.pt")
    if os.path.exists(final_checkpoint_path):
        print(f"✅ Final checkpoint saved successfully: {final_checkpoint_path}")
        inspect_checkpoint(final_checkpoint_path)
    else:
        print(f"❌ Final checkpoint NOT found: {final_checkpoint_path}")
        return False
    
    print()
    print("🎉 Checkpoint functionality test completed!")
    print("=" * 50)
    
    # List all created checkpoints
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        print(f"📋 Created checkpoints: {checkpoints}")
    
    return True

if __name__ == "__main__":
    success = test_checkpoint_functionality()
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Tests failed!")
        exit(1) 