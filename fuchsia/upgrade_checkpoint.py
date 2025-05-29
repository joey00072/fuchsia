#!/usr/bin/env python3
"""
Upgrade old checkpoints to include the missing dataset_position and config fields
"""

import torch
import os
import sys
import datetime
from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from rich import box

console = Console()

def create_status_panel(title: str, message: str, style: str = "white") -> Panel:
    """Create a beautiful status panel"""
    return Panel(
        Align.center(Text(message, style=style)),
        title=title,
        box=box.ROUNDED,
        border_style=style
    )

def upgrade_checkpoint(checkpoint_path: str, output_path: str = None):
    """Upgrade a checkpoint to include missing fields"""
    
    if not os.path.exists(checkpoint_path):
        console.print(create_status_panel("❌ Error", f"Checkpoint file not found: {checkpoint_path}", "red"))
        return False
    
    if output_path is None:
        output_path = checkpoint_path  # Overwrite the original
    
    try:
        console.print(create_status_panel("📂 Loading", f"Loading checkpoint: {checkpoint_path}", "blue"))
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if not isinstance(checkpoint, dict):
            console.print(create_status_panel("❌ Error", "Checkpoint is not a dictionary", "red"))
            return False
        
        # Check what's missing
        missing_fields = []
        
        # Add dataset_position if missing
        if "dataset_position" not in checkpoint:
            iteration = checkpoint.get("iteration", 0)
            checkpoint["dataset_position"] = iteration  # Assume dataset position equals iteration
            missing_fields.append(f"dataset_position (set to {iteration})")
        
        # Add config if missing
        if "config" not in checkpoint:
            checkpoint["config"] = {
                "model_name": "unknown",  # Will need to be updated manually if needed
                "learning_rate": 0.0005,  # Default value
                "group_size": 8,
                "batch_size": 4,
                "max_iterations": 1000,
            }
            missing_fields.append("config (added default values)")
        
        # Add timestamp if missing
        if "timestamp" not in checkpoint:
            checkpoint["timestamp"] = datetime.datetime.now().isoformat()
            missing_fields.append("timestamp (current time)")
        
        if missing_fields:
            console.print(create_status_panel("🔧 Upgrading", f"Adding missing fields: {', '.join(missing_fields)}", "yellow"))
            
            # Save the upgraded checkpoint
            console.print(create_status_panel("💾 Saving", f"Saving upgraded checkpoint to: {output_path}", "blue"))
            torch.save(checkpoint, output_path)
            
            console.print(create_status_panel("✅ Success", "Checkpoint upgraded successfully!", "green"))
        else:
            console.print(create_status_panel("✅ Already Good", "Checkpoint already has all required fields", "green"))
        
        return True
        
    except Exception as e:
        console.print(create_status_panel("❌ Error", f"Failed to upgrade checkpoint: {str(e)}", "red"))
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print(create_status_panel("Usage", "python upgrade_checkpoint.py <checkpoint_path> [output_path]", "blue"))
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    console.print(Panel(
        Align.center(Text("🔧 Checkpoint Upgrader", style="bold magenta")),
        title="GRPO Checkpoint Upgrade Tool",
        box=box.ROUNDED,
        border_style="magenta"
    ))
    console.print()
    
    success = upgrade_checkpoint(checkpoint_path, output_path)
    sys.exit(0 if success else 1) 