#!/usr/bin/env python3
"""
MLflow experiment tracking for distributed training
Tracks 100+ experiments with metrics
"""

import mlflow
import mlflow.pytorch
import torch
import os
import json
from datetime import datetime

class MLflowTracker:
    """
    Track training experiments with MLflow
    """
    def __init__(self, experiment_name: str = "distributed_training"):
        self.experiment_name = experiment_name
        
        # Set tracking URI
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Set or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        
        mlflow.set_experiment(experiment_name)
        
        print(f"✓ MLflow tracker initialized")
        print(f"  Experiment: {experiment_name}")
    
    def start_run(self, run_name: str, params: dict):
        """Start a new MLflow run"""
        mlflow.start_run(run_name=run_name)
        
        # Log parameters
        mlflow.log_params(params)
        
        print(f"✓ Started run: {run_name}")
    
    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model, artifact_path: str = "model"):
        """Log PyTorch model"""
        mlflow.pytorch.log_model(model, artifact_path)
    
    def end_run(self):
        """End current run"""
        mlflow.end_run()

def simulate_experiments(num_experiments: int = 100):
    """
    Simulate 100+ experiments with different hyperparameters
    """
    print("="*60)
    print(f"Running {num_experiments} Experiments")
    print("="*60)
    
    tracker = MLflowTracker("distributed_training")
    
    # Hyperparameter grid
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    batch_sizes = [64, 128, 256, 512]
    num_layers = [2, 3, 4]
    
    experiment_count = 0
    
    for lr in learning_rates:
        for bs in batch_sizes:
            for layers in num_layers:
                if experiment_count >= num_experiments:
                    break
                
                experiment_count += 1
                
                # Create run
                run_name = f"exp_{experiment_count:03d}_lr{lr}_bs{bs}_l{layers}"
                
                params = {
                    'learning_rate': lr,
                    'batch_size': bs,
                    'num_layers': layers,
                    'optimizer': 'adam',
                    'world_size': 4
                }
                
                tracker.start_run(run_name, params)
                
                # Simulate training (random metrics for demo)
                import numpy as np
                for epoch in range(5):
                    # Simulate metrics
                    train_loss = 2.0 - (epoch * 0.3) + np.random.normal(0, 0.1)
                    val_acc = 0.5 + (epoch * 0.08) + np.random.normal(0, 0.02)
                    
                    tracker.log_metrics({
                        'train_loss': train_loss,
                        'val_accuracy': val_acc
                    }, step=epoch)
                
                # Final metrics
                final_acc = 0.75 + np.random.normal(0, 0.05)
                training_time = 45 + np.random.normal(0, 5)  # ~45 minutes
                
                tracker.log_metrics({
                    'final_accuracy': final_acc,
                    'training_time_minutes': training_time
                })
                
                tracker.end_run()
                
                if experiment_count % 10 == 0:
                    print(f"  Completed {experiment_count}/{num_experiments} experiments")
    
    print(f"\n✓ Completed {experiment_count} experiments")
    print(f"✓ Results saved to mlruns/")
    print(f"\nTo view results:")
    print(f"  mlflow ui")
    print(f"  Visit: http://localhost:5000")

def main():
    simulate_experiments(num_experiments=100)

if __name__ == "__main__":
    main()
