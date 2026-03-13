#!/usr/bin/env python3
"""
Model configuration management
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for model training"""
    model_name: str = "resnet18"
    num_classes: int = 10
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 10
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    momentum: float = 0.9
    use_mixed_precision: bool = False
    gradient_clip: Optional[float] = None

def get_default_config() -> ModelConfig:
    """Get default configuration"""
    return ModelConfig()

def main():
    config = get_default_config()
    print(f"Default config: {config}")

if __name__ == "__main__":
    main()
