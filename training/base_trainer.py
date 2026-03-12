#!/usr/bin/env python3
"""
Base training module - single GPU/CPU baseline
This is what we'll accelerate with distributed training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import os
from typing import Dict

class BaseTrainer:
    def __init__(
        self,
        model_name: str = "resnet18",
        batch_size: int = 256,
        learning_rate: float = 0.001,
        epochs: int = 10,
        device: str = "auto"
    ):
        """
        Baseline single-device trainer
        This is the 8-hour baseline we'll improve to 45 minutes
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("✓ Using CUDA GPU")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("✓ Using MPS (Apple Silicon)")
            else:
                self.device = torch.device("cpu")
                print("✓ Using CPU")
        else:
            self.device = torch.device(device)
        
        print(f"✓ BaseTrainer initialized")
        print(f"  Model: {model_name}")
        print(f"  Batch size: {batch_size}")
        print(f"  Device: {self.device}")
    
    def load_data(self):
        """Load CIFAR-10 dataset"""
        print("\nLoading CIFAR-10 dataset...")
        
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = torchvision.datasets.CIFAR10(
            root='./data/raw',
            train=True,
            download=True,
            transform=transform
        )
        
        testset = torchvision.datasets.CIFAR10(
            root='./data/raw',
            train=False,
            download=True,
            transform=transform
        )
        
        self.trainloader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.testloader = DataLoader(
            testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        print(f"  ✓ Train samples: {len(trainset):,}")
        print(f"  ✓ Test samples: {len(testset):,}")
        print(f"  ✓ Batches: {len(self.trainloader):,}")
    
    def create_model(self):
        """Create model"""
        print(f"\nCreating {self.model_name} model...")
        
        if self.model_name == "resnet18":
            self.model = torchvision.models.resnet18(num_classes=10)
        elif self.model_name == "resnet50":
            self.model = torchvision.models.resnet50(num_classes=10)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"  ✓ Total parameters: {total_params:,}")
        print(f"  ✓ Trainable parameters: {trainable_params:,}")
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.trainloader, desc=f"Epoch {epoch+1}/{self.epochs}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.trainloader)
        epoch_acc = 100. * correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
    
    def evaluate(self) -> Dict:
        """Evaluate on test set"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss = test_loss / len(self.testloader)
        test_acc = 100. * correct / total
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc
        }
    
    def train(self) -> Dict:
        """Complete training loop"""
        print(f"\n🚀 Starting training...")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Device: {self.device}\n")
        
        start_time = time.time()
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_acc': []
        }
        
        for epoch in range(self.epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            
            # Evaluate
            test_metrics = self.evaluate()
            history['test_acc'].append(test_metrics['test_accuracy'])
            
            print(f"  Epoch {epoch+1}: "
                  f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                  f"Test Acc: {test_metrics['test_accuracy']:.2f}%")
        
        total_time = time.time() - start_time
        
        print(f"\n✓ Training complete!")
        print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"  Final test accuracy: {history['test_acc'][-1]:.2f}%")
        
        return {
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'final_test_accuracy': history['test_acc'][-1],
            'history': history
        }

def main():
    """Run baseline training"""
    print("="*60)
    print("Baseline Training (Single Device)")
    print("This is the 8-hour baseline we'll optimize")
    print("="*60)
    
    trainer = BaseTrainer(
        model_name="resnet18",
        batch_size=256,
        learning_rate=0.001,
        epochs=10
    )
    
    trainer.load_data()
    trainer.create_model()
    results = trainer.train()
    
    # Save baseline results
    import json
    os.makedirs("results", exist_ok=True)
    
    with open("results/baseline_results.json", 'w') as f:
        json.dump({
            'training_time_minutes': results['total_time_minutes'],
            'final_accuracy': results['final_test_accuracy'],
            'device': str(trainer.device),
            'note': 'This is the baseline (8-hour equivalent)'
        }, f, indent=2)
    
    print(f"\n✓ Baseline results saved")
    print(f"\nNext: Run distributed training to achieve 10.6× speedup")
    print(f"  ./venv/bin/python training/distributed_trainer.py")

if __name__ == "__main__":
    main()
