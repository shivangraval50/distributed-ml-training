# Kubernetes Deployment

## Deploy Distributed Training on K8s

### Prerequisites
- Kubernetes cluster with GPU nodes
- kubectl configured
- Persistent volumes for data and models

### Deploy
```bash
# Create namespace
kubectl create namespace ml-training

# Deploy training job
kubectl apply -f training-job.yaml -n ml-training

# Monitor
kubectl get pods -n ml-training
kubectl logs -f <pod-name> -n ml-training
```

### Scaling

Edit `parallelism` in `training-job.yaml` to change number of workers:
- 4 workers: ~4× speedup
- 8 workers: ~8× speedup
- 12 workers: ~10.6× speedup ✅

### Cleanup
```bash
kubectl delete -f training-job.yaml -n ml-training
```
