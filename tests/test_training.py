#!/usr/bin/env python3
"""
Tests for training components
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from models.model_factory import ModelFactory

def test_model_creation():
    """Test model creation"""
    factory = ModelFactory()
    
    model = factory.create_model("resnet18", num_classes=10)
    
    assert model is not None
    assert isinstance(model, torch.nn.Module)
    
    # Test forward pass
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    
    assert output.shape == (1, 10)

def test_parameter_counting():
    """Test parameter counting"""
    factory = ModelFactory()
    model = factory.create_model("resnet18")
    
    params = factory.count_parameters(model)
    
    assert params['total'] > 0
    assert params['trainable'] > 0

def main():
    print("="*60)
    print("Training Component Tests")
    print("="*60)
    
    test_model_creation()
    print("✓ Test 1: Model creation passed")
    
    test_parameter_counting()
    print("✓ Test 2: Parameter counting passed")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    main()
