# Persistent Storage Setup for ModelExpress

This guide explains how to set up persistent storage for caching downloaded models in ModelExpress.

## Overview

ModelExpress downloads models from Hugging Face Hub and caches them locally. To ensure models persist across container restarts and deployments, you need to configure persistent storage.

## Storage Locations

- **Model Cache**: Hugging Face Hub cache directory (default: `~/.cache/huggingface/hub/`)
- **Database**: SQLite database file (`models.db`)

## 1. Docker Compose Setup (Recommended)

### Basic Setup
```bash
# Start with persistent volumes
docker-compose up -d
```

The `docker-compose.yml` file is already configured with:
- `model-cache` volume for Hugging Face models
- `model-db` volume for the SQLite database
- `HF_HUB_CACHE=/app/models` environment variable

### Custom Storage Location
```yaml
# docker-compose.override.yml
version: '3'
services:
  model-express-server:
    volumes:
      - /path/to/your/models:/app/models
      - /path/to/your/database:/app/models.db
```

## 2. Docker Run Setup

### Using Named Volumes
```bash
# Create volumes
docker volume create model-express-cache
docker volume create model-express-db

# Run container
docker run -d \
  --name model-express-server \
  -p 8000:8000 \
  -e HF_HUB_CACHE=/app/models \
  -v model-express-cache:/app/models \
  -v model-express-db:/app/models.db \
  model-express:latest
```

### Using Bind Mounts
```bash
# Create local directories
mkdir -p ~/model-express-cache
mkdir -p ~/model-express-db

# Run container
docker run -d \
  --name model-express-server \
  -p 8000:8000 \
  -e HF_HUB_CACHE=/app/models \
  -v ~/model-express-cache:/app/models \
  -v ~/model-express-db:/app/models.db \
  model-express:latest
```

## 3. Kubernetes Setup

### Deploy with Persistent Volumes
```bash
# Apply the deployment (includes PVCs)
kubectl apply -f k8s-deployment.yaml

# Check PVC status
kubectl get pvc
kubectl get pv
```

### Custom Storage Class
```yaml
# custom-storage.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd
```

## 4. Local Development Setup

### Environment Variables
```bash
# Set custom cache location
export HF_HUB_CACHE=/path/to/your/models

# Run server
cargo run --bin model_express_server
```

### Docker for Local Development
```bash
# Create development volumes
docker volume create dev-model-cache
docker volume create dev-model-db

# Run with development configuration
docker run -d \
  --name model-express-dev \
  -p 8001:8000 \
  -e HF_HUB_CACHE=/app/models \
  -e LOG_LEVEL=debug \
  -v dev-model-cache:/app/models \
  -v dev-model-db:/app/models.db \
  model-express:latest
```

## 5. Storage Sizing Guidelines

### Model Cache Size
- **Small models** (BERT, T5-small): 100MB - 500MB each
- **Medium models** (GPT-2, BART): 500MB - 2GB each
- **Large models** (GPT-3, T5-large): 2GB - 10GB each
- **Extra large models** (GPT-4, LLaMA): 10GB+ each

### Recommended Storage
- **Development**: 10GB
- **Production (few models)**: 50GB
- **Production (many models)**: 100GB+
- **Database**: 1GB (sufficient for metadata)

## 6. Backup and Migration

### Backup Models
```bash
# Backup model cache
docker run --rm -v model-express-cache:/cache -v $(pwd):/backup alpine tar czf /backup/model-cache-backup.tar.gz -C /cache .

# Backup database
docker run --rm -v model-express-db:/db -v $(pwd):/backup alpine cp /db/models.db /backup/
```

### Restore Models
```bash
# Restore model cache
docker run --rm -v model-express-cache:/cache -v $(pwd):/backup alpine tar xzf /backup/model-cache-backup.tar.gz -C /cache

# Restore database
docker run --rm -v model-express-db:/db -v $(pwd):/backup alpine cp /backup/models.db /db/
```

## 7. Monitoring Storage Usage

### Check Volume Usage
```bash
# Docker volumes
docker system df -v

# Kubernetes
kubectl exec -it deployment/model-express-server -- du -sh /app/models
kubectl exec -it deployment/model-express-server -- ls -la /app/models.db
```

### Clean Up Old Models
```bash
# List downloaded models
kubectl exec -it deployment/model-express-server -- ls -la /app/models

# Remove specific model (be careful!)
kubectl exec -it deployment/model-express-server -- rm -rf /app/models/models--google--t5-small
```

## 8. Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   # Fix permissions
   docker run --rm -v model-express-cache:/cache alpine chown -R 1000:1000 /cache
   ```

2. **Storage Full**
   ```bash
   # Check usage
   docker system df
   
   # Clean up
   docker system prune -a
   ```

3. **Database Corruption**
   ```bash
   # Backup and recreate
   docker run --rm -v model-express-db:/db alpine cp /db/models.db /db/models.db.backup
   ```

### Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HUB_CACHE` | `~/.cache/huggingface/hub/` | Hugging Face cache directory |
| `SERVER_PORT` | `8000` | Server port |
| `LOG_LEVEL` | `info` | Logging level |

## 9. Security Considerations

- **Network Storage**: Use encrypted volumes for sensitive models
- **Access Control**: Restrict volume access to necessary users
- **Backup Encryption**: Encrypt backups of model cache
- **Audit Logging**: Monitor model downloads and usage

## 10. Performance Optimization

- **SSD Storage**: Use SSD storage for better I/O performance
- **Network Storage**: Consider NFS or cloud storage for shared access
- **Compression**: Models are already compressed, avoid additional compression
- **Caching**: Enable filesystem caching for frequently accessed models 