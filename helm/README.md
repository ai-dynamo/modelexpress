# ModelExpress Helm Chart

This Helm chart deploys ModelExpress, a model serving and management platform, to Kubernetes.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- Access to NVIDIA Container Registry (nvcr.io) for pulling the ModelExpress image

## Installation

### 1. Create NVIDIA Container Registry Secret

Before installing the chart, you must create a Kubernetes secret to access the private NVIDIA Container Registry (nvcr.io). The default image requires authentication.

```bash
# Create the secret in your target namespace
kubectl create secret docker-registry nvcr-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password='YOUR_NVCR_API_KEY' \
  --docker-email='your-email@nvidia.com' \
  --namespace=your-namespace

# Or create it in the default namespace if you plan to use that
kubectl create secret docker-registry nvcr-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password='YOUR_NVCR_API_KEY' \
  --docker-email='your-email@nvidia.com'
```

**Important Notes:**
- Replace `YOUR_NVCR_API_KEY` with your actual NVIDIA Container Registry API key
- The username must be `$oauthtoken` (literal string, not a variable)
- The email should be your NVIDIA email address
- The secret name `nvcr-secret` is referenced in the default values

### 2. Update Values to Use the Secret

The default `values.yaml` already includes the image pull secret configuration:

```yaml
# values.yaml (default)
imagePullSecrets:
  - name: nvcr-secret
```

If you're using a custom values file, ensure it includes this configuration or the deployment will fail with image pull errors.

### 3. Add the Helm repository (if using a repository)

```bash
helm repo add modelexpress https://your-repo-url
helm repo update
```

### 4. Install the chart

```bash
# Install with default values
helm install my-modelexpress ./helm

# Install with custom values
helm install my-modelexpress ./helm -f values.yaml

# Install in a specific namespace
helm install my-modelexpress ./helm --namespace modelexpress --create-namespace
```

## Configuration

### ⚠️ Important: Override Production Values

**CRITICAL:** The `values-production.yaml` file contains example values that **MUST** be overridden for your environment:

- **Domain Names**: `modelexpress.yourdomain.com` is a placeholder - replace with your actual domain
- **TLS Certificates**: The TLS configuration references `modelexpress-tls` secret - ensure this exists or update the configuration
- **Storage Classes**: `fast-ssd` storage class may not exist in your cluster - verify or change to an available storage class
- **Node Selectors**: `node-type: "compute"` and tolerations may not match your cluster setup

**Always review and customize production values before deployment:**

```bash
# Copy and customize production values
cp helm/values-production.yaml helm/my-production-values.yaml
# Edit my-production-values.yaml with your actual values
helm install modelexpress ./helm -f helm/my-production-values.yaml
```

The following table lists the configurable parameters of the ModelExpress chart and their default values.

| Parameter                                    | Description                                    | Default |
|----------------------------------------------|------------------------------------------------|---------|
| `replicaCount`                               | Number of ModelExpress replicas                | `1`     |
| `image.repository`                           | ModelExpress image repository                  | `nvcr.io/nvidia/ai-dynamo/modelexpress-server` |
| `image.pullPolicy`                           | Image pull policy                              | `IfNotPresent` |
| `image.tag`                                  | ModelExpress image tag                         | `0.2.0` |
| `imagePullSecrets`                           | Image pull secrets for nvcr.io access          | `[]`     |
| `nameOverride`                               | Override the chart name                        | `""`     |
| `fullnameOverride`                           | Override the full app name                     | `""`     |
| `serviceAccount.create`                      | Create a service account                       | `true`   |
| `serviceAccount.annotations`                 | Service account annotations                    | `{}`     |
| `serviceAccount.name`                        | Service account name                           | `""`     |
| `podAnnotations`                             | Pod annotations                                | `{}`     |
| `podSecurityContext`                         | Pod security context                           | `{}`     |
| `securityContext`                            | Container security context                     | `{}`     |
| `service.type`                               | Service type                                   | `ClusterIP` |
| `service.port`                               | Service port                                   | `8001`   |
| `ingress.enabled`                            | Enable ingress                                 | `false`  |
| `ingress.className`                          | Ingress class name                             | `""`     |
| `ingress.annotations`                        | Ingress annotations                            | `{}`     |
| `ingress.hosts`                              | Ingress hosts                                  | `[]`     |
| `ingress.tls`                                | Ingress TLS configuration                      | `[]`     |
| `resources.limits.cpu`                       | CPU limit                                      | `500m`   |
| `resources.limits.memory`                    | Memory limit                                   | `256Mi`  |
| `resources.requests.cpu`                     | CPU request                                    | `200m`   |
| `resources.requests.memory`                  | Memory request                                 | `128Mi`  |
| `persistence.enabled`                        | Enable persistence                             | `true`   |
| `persistence.storageClass`                   | Storage class                                  | `""`     |
| `persistence.accessMode`                     | Access mode                                    | `ReadWriteOnce` |
| `persistence.size`                           | Storage size                                   | `10Gi`   |
| `persistence.mountPath`                      | Mount path                                     | `/root`  |
| `env.MODEL_EXPRESS_SERVER_PORT`              | Server port                                    | `8001`   |
| `env.MODEL_EXPRESS_LOGGING_LEVEL`            | Logging level                                  | `info`   |
| `env.MODEL_EXPRESS_DATABASE_PATH`            | Database path                                  | `/app/models.db` |
| `env.MODEL_EXPRESS_CACHE_DIRECTORY`          | Cache directory                                | `/app/cache` |
| `livenessProbe.enabled`                      | Enable liveness probe                          | `true`   |
| `readinessProbe.enabled`                     | Enable readiness probe                         | `true`   |
| `nodeSelector`                               | Node selector                                  | `{}`     |
| `tolerations`                                | Tolerations                                    | `[]`     |
| `affinity`                                   | Affinity rules                                 | `{}`     |

## Examples

### Basic Installation

```bash
helm install modelexpress ./helm
```

### Custom Image Repository

```yaml
# values.yaml
image:
  repository: your-registry/modelexpress-server
  tag: v1.0.0
  pullPolicy: Always
```

### With Ingress

**⚠️ Warning:** Replace `modelexpress.example.com` with your actual domain and ensure the TLS secret exists.

```yaml
# values.yaml
ingress:
  enabled: true
  className: nginx
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: modelexpress.example.com  # ← Replace with your actual domain
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: modelexpress-tls  # ← Ensure this secret exists
      hosts:
        - modelexpress.example.com  # ← Replace with your actual domain
```

### With Custom Resources

```yaml
# values.yaml
resources:
  limits:
    cpu: 1000m
    memory: 1Gi
  requests:
    cpu: 500m
    memory: 512Mi
```

### With Custom Storage

```yaml
# values.yaml
persistence:
  enabled: true
  storageClass: fast-ssd
  size: 50Gi
  mountPath: /app/data
```

### With Additional Environment Variables

```yaml
# values.yaml
extraEnv:
  - name: CUSTOM_VAR
    value: "custom_value"
  - name: SECRET_VAR
    valueFrom:
      secretKeyRef:
        name: modelexpress-secrets
        key: secret-key
```

## Upgrading

```bash
helm upgrade my-modelexpress ./helm
```

## Uninstalling

```bash
helm uninstall my-modelexpress
```

## Troubleshooting

### Check Pod Status

```bash
kubectl get pods -l app.kubernetes.io/name=modelexpress
```

### Check Logs

```bash
kubectl logs -l app.kubernetes.io/name=modelexpress
```

### Check Service

```bash
kubectl get svc -l app.kubernetes.io/name=modelexpress
```

### Port Forward for Local Access

```bash
kubectl port-forward svc/my-modelexpress 8001:8001
```

### Image Pull Issues

If you encounter `ErrImagePull` or `ImagePullBackOff` errors:

1. **Check if the nvcr.io secret exists:**
   ```bash
   kubectl get secrets -n your-namespace | grep nvcr
   ```

2. **Verify the secret is properly configured:**
   ```bash
   kubectl describe secret nvcr-secret -n your-namespace
   ```

3. **Check if the secret is referenced in your values:**
   ```yaml
   imagePullSecrets:
     - name: nvcr-secret
   ```

4. **Verify your API key is correct:**
   ```bash
   # Test Docker login locally
   docker login nvcr.io -u '$oauthtoken' -p 'YOUR_NVCR_API_KEY'
   ```

5. **Check pod events for detailed error messages:**
   ```bash
   kubectl describe pod -l app.kubernetes.io/name=modelexpress -n your-namespace
   ```

## Using the Official Image

The Helm chart uses the official NVIDIA ModelExpress image from the NVIDIA Container Registry (nvcr.io):

```bash
# Login to nvcr.io (requires NVIDIA credentials)
docker login nvcr.io -u '$oauthtoken' -p 'YOUR_NVCR_API_KEY'

# Pull the image
docker pull nvcr.io/nvidia/ai-dynamo/modelexpress-server:0.2.0
```

**Note:** The default image requires authentication. See the [Installation](#installation) section for creating the required Kubernetes secret.

## Contributing

When contributing to this Helm chart, please ensure:

1. All templates follow Helm best practices
2. Values are properly documented
3. Examples are provided for common use cases
4. Tests are included for the chart
