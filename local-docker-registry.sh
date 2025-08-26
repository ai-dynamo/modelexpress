#!/bin/bash
# Setup local Docker registry for Kubernetes deployment
set -e
echo "Setting up local Docker registry for Kubernetes..."
# 1. Start the local registry
echo "Starting local Docker registry..."
docker compose -f docker-compose-registry.yml up -d registry
# 2. Wait for registry to be ready
echo "Waiting for registry to be ready..."
sleep 5
# 3. Build the model-express image
echo "Building model-express image..."
docker build -t model-express:latest .
# 4. Tag and push the model-express image to local registry
echo "Tagging and pushing model-express image to local registry..."
docker tag model-express:latest localhost:5001/model-express:latest
docker push localhost:5001/model-express:latest
# 5. Configure Kubernetes to use local registry
echo "Configuring Kubernetes to use local registry..."
# Create a secret for the local registry (if needed)
kubectl create secret docker-registry local-registry-secret \
  --docker-server=localhost:5001 \
  --docker-username="" \
  --docker-password="" \
  --docker-email="" \
  --dry-run=client -o yaml | kubectl apply -f -
# 6. Update the deployment to use the local registry
echo "Updating deployment to use local registry..."
# Check if we're running on a single-node cluster (like kind or minikube)
if kubectl get nodes | grep -q "kind-control-plane\|minikube"; then
    echo "Detected single-node cluster, updating deployment..."
    # For single-node clusters, we can use localhost
    kubectl apply -f k8s-deployment-local.yaml
else
    echo "Multi-node cluster detected. You may need to:"
    echo "1. Load the image on all nodes using:"
    echo "   ./load-image-on-nodes.sh"
    echo "2. Update the deployment to use the correct registry URL"
fi
echo "Local registry setup complete!"
echo ""
echo "To deploy to Kubernetes:"
echo "kubectl apply -f k8s-deployment-local.yaml"
echo ""
echo "To check the deployment:"
echo "kubectl get pods"
echo "kubectl logs -f deployment/model-express-server"