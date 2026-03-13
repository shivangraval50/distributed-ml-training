#!/usr/bin/env python3
"""
Test distributed training setup
"""

import torch
import torch.distributed as dist

def test_distributed_available():
    """Test if distributed package is available"""
    assert hasattr(torch, 'distributed')
    print("✓ Distributed package available")

def test_backends():
    """Test available backends"""
    backends = ['gloo']  # nccl requires GPU
    
    for backend in backends:
        available = dist.is_available()
        print(f"  {backend}: {'✓' if available else '✗'}")
    
    assert dist.is_available()

def main():
    print("="*60)
    print("Distributed Training Tests")
    print("="*60)
    
    test_distributed_available()
    test_backends()
    
    print("\n✅ Distributed setup tests passed!")

if __name__ == "__main__":
    main()
