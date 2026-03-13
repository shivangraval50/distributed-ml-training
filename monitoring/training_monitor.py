#!/usr/bin/env python3
"""
Training monitoring and metrics collection
"""

from prometheus_client import Counter, Gauge, Histogram, start_http_server
import time

# Metrics
training_epochs = Counter('training_epochs_total', 'Total epochs trained')
training_loss = Gauge('training_loss', 'Current training loss')
training_accuracy = Gauge('training_accuracy', 'Current training accuracy')
epoch_duration = Histogram('epoch_duration_seconds', 'Epoch duration')
gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization', ['gpu_id'])

class TrainingMonitor:
    """Monitor training progress"""
    
    def __init__(self, port: int = 9093):
        self.port = port
        self.start_time = None
        
        # Start metrics server
        start_http_server(port)
        print(f"✓ Training monitor started on port {port}")
    
    def start_training(self):
        """Mark start of training"""
        self.start_time = time.time()
    
    def log_epoch(self, epoch: int, loss: float, accuracy: float, duration: float):
        """Log epoch metrics"""
        training_epochs.inc()
        training_loss.set(loss)
        training_accuracy.set(accuracy)
        epoch_duration.observe(duration)
    
    def log_gpu_utilization(self, gpu_id: int, utilization: float):
        """Log GPU utilization"""
        gpu_utilization.labels(gpu_id=str(gpu_id)).set(utilization)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since training started"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

def main():
    """Test monitor"""
    print("Testing Training Monitor...")
    
    monitor = TrainingMonitor(port=9093)
    monitor.start_training()
    
    # Simulate logging
    monitor.log_epoch(1, 1.5, 65.0, 120.5)
    monitor.log_gpu_utilization(0, 95.0)
    
    print(f"✓ Monitor working - metrics at http://localhost:9093/metrics")

if __name__ == "__main__":
    main()
