#!/usr/bin/env python3
"""
Model factory for creating different architectures
"""

import torch
import torch.nn as nn
import torchvision.models as models

class ModelFactory:
    """Factory for creating different model architectures"""
    
    @staticmethod
    def create_model(model_name: str, num_classes: int = 10):
        """
        Create model by name
        
        Args:
            model_name: resnet18, resnet50, efficientnet_b0, etc.
            num_classes: Number of output classes
        """
        if model_name == "resnet18":
            model = models.resnet18(pretrained=False, num_classes=num_classes)
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=False, num_classes=num_classes)
        elif model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=False, num_classes=num_classes)
        elif model_name == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=False, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model
    
    @staticmethod
    def count_parameters(model):
        """Count model parameters"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable
        }

def main():
    """Test model factory"""
    print("="*60)
    print("Model Factory Test")
    print("="*60)
    
    factory = ModelFactory()
    
    # Create different models
    models_to_test = ['resnet18', 'resnet50']
    
    for model_name in models_to_test:
        print(f"\nCreating {model_name}...")
        model = factory.create_model(model_name, num_classes=10)
        params = factory.count_parameters(model)
        
        print(f"  ✓ Created {model_name}")
        print(f"  ✓ Parameters: {params['total']:,}")
    
    print(f"\n✓ Model factory working correctly")

if __name__ == "__main__":
    main()
