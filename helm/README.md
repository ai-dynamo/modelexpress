<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ModelExpress Helm Chart

This Helm chart deploys ModelExpress, a model serving and management platform, to Kubernetes. For the broader deployment guide covering Docker, standalone K8s, and P2P transfers, see [`docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md).

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+

## Installation

### 1. Add the Helm repository (if using a repository)

```bash
helm repo add modelexpress https://your-repo-url
helm repo update
```

### 2. Install or update CRDs for the Kubernetes backend

The chart ships the `ModelMetadata` and `ModelCacheEntry` CRDs in its `crds/`
directory. Helm installs CRDs that are missing during `helm install`, but it does
not update CRDs that already exist during either `helm install` or
`helm upgrade`. Because CRDs are cluster-scoped, an older definition can remain
even when installing a new release in a different namespace.

Before upgrading the chart, or installing it on a cluster that already has
ModelExpress CRDs, apply the definitions from the chart with cluster-admin
credentials:

```bash
kubectl apply -f helm/crds/modelexpress-crds.yaml
```

When using a packaged or remote chart rather than a source checkout, extract and
apply the CRDs from the same chart version you are about to install:

```bash
helm show crds CHART_REFERENCE --version CHART_VERSION | kubectl apply -f -
```

This step is required to deliver schema changes to existing clusters. Helm does
not remove these CRDs when the application release is uninstalled.

### 3. Install the chart

To view the available tags for the official ModelExpress image, see the
[ModelExpress Server tags](https://catalog.ngc.nvidia.com/orgs/nvidia/ai-dynamo/containers/modelexpress-server/-/tags)
in the NVIDIA NGC catalog.

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
| `image.tag`                                  | ModelExpress image tag                         | Chart `appVersion` |
| `imagePullSecrets`                           | Image pull secrets for nvcr.io access          | `[]`     |
| `nameOverride`                               | Override the chart name                        | `""`     |
| `fullnameOverride`                           | Override the full app name                     | `""`     |
| `serviceAccount.create`                      | Create a service account                       | `true`   |
| `serviceAccount.annotations`                 | Service account annotations                    | `{}`     |
| `serviceAccount.name`                        | Service account name                           | `""`     |
| `serviceAccount.rbac.enabled`                | Create a ClusterRole and ClusterRoleBinding for the Kubernetes metadata backend | `false` |
| `podAnnotations`                             | Pod annotations                                | `{}`     |
| `podSecurityContext`                         | Pod security context                           | `{runAsNonRoot: true, runAsUser: 1000, runAsGroup: 1000, fsGroup: 1000}` |
| `securityContext`                            | Container security context                     | `{runAsNonRoot: true}` |
| `service.type`                               | Service type                                   | `ClusterIP` |
| `service.port`                               | Service port                                   | `8001`   |
| `metrics.enabled`                            | Serve Prometheus metrics on their own port     | `true`   |
| `metrics.port`                               | Metrics port. Single source of truth for the env var, containerPort, annotation and Service port | `9401` |
| `metrics.podAnnotations`                     | Emit `prometheus.io/{scrape,port,path}`. Inert on Prometheus Operator clusters -- use `metrics.podMonitor` there | `true` |
| `metrics.service`                            | Publish the metrics port on the Service too    | `false`  |
| `metrics.podMonitor.enabled`                 | PodMonitor for the server. Requires the Prometheus Operator CRDs | `false` |
| `metrics.podMonitor.additionalLabels`        | Labels the Prometheus `podMonitorSelector` matches. Usually `release: <your-prometheus-release>`, without which it is silently ignored | `{}` |
| `metrics.podMonitor.interval`                | Scrape interval                                | `30s`    |
| `metrics.podMonitor.scrapeTimeout`           | Scrape timeout                                 | `10s`    |
| `metrics.podMonitor.relabelings`             | Passed through to the endpoint's `relabelings` | `[]`     |
| `metrics.podMonitor.metricRelabelings`       | Passed through to the endpoint's `metricRelabelings` | `[]` |
| `metrics.clientPodMonitor.enabled`           | PodMonitor for client metrics in your inference pods, which this chart does not deploy | `false` |
| `metrics.clientPodMonitor.selector`          | Label selector for those pods. Required -- an empty selector matches every pod in the namespace | `{}` |
| `metrics.clientPodMonitor.portName`          | Named port on the engine pod. Must match the name in YOUR manifest — the default is the one the shipped example uses, and a mismatch renders fine then matches no port | `mx-metrics` |
| `metrics.clientPodMonitor.namespaceSelector` | Namespaces to search. Empty means the release namespace | `{}` |
| `metrics.clientPodMonitor.interval`          | Scrape interval                                | `30s`    |
| `metrics.clientPodMonitor.scrapeTimeout`     | Scrape timeout                                 | `10s`    |
| `metrics.clientPodMonitor.additionalLabels`  | Labels the Prometheus `podMonitorSelector` matches. Same caveat as `metrics.podMonitor.additionalLabels` | `{}` |
| `metrics.clientPodMonitor.relabelings`       | Passed through to the endpoint's `relabelings` | `[]`     |
| `metrics.clientPodMonitor.metricRelabelings` | Passed through to the endpoint's `metricRelabelings` | `[]` |
| `metrics.dashboard.additionalLabels`         | Extra labels on the dashboard ConfigMap        | `{}`     |
| `metrics.rules.enabled`                      | Ship alerting rules as a PrometheusRule        | `false`  |
| `metrics.rules.client`                       | Also include client alerts, which need client metrics enabled and P2P in use | `false` |
| `metrics.rules.additionalLabels`             | Labels the Prometheus `ruleSelector` matches. Same caveat as the PodMonitor | `{}` |
| `metrics.rules.thresholds`                   | Per-alert thresholds. See `values.yaml`        | see values |
| `metrics.dashboard.enabled`                  | Ship the Grafana dashboard as a ConfigMap      | `false`  |
| `metrics.dashboard.label`                    | Grafana sidecar discovery label                | `grafana_dashboard` |
| `metrics.dashboard.labelValue`               | Value for that label                           | `"1"`    |
| `metrics.{podMonitor,clientPodMonitor,rules}.skipApiVersionCheck` | Render the resource even when the `monitoring.coreos.com/v1` API is not detected. For CRDs applied in the same operation as this chart; otherwise enabling one without the Operator is an install-time failure rather than a silent skip | `false` |
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
| `env.MODEL_EXPRESS_LOG_LEVEL`                | Logging level                                  | `info`   |
| `env.MODEL_EXPRESS_CACHE_DIRECTORY`          | Cache directory                                | `/root`  |
| `env.MX_METADATA_BACKEND`                    | Distributed backend (`redis` or `kubernetes`). Server fails to start without this. | `<required>` |
| `env.REDIS_URL`                              | Redis connection URL; required when backend is `redis`. Chart does not bundle Redis. | `<required when backend=redis>` |
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

### With Kubernetes Backend RBAC

Enabling `serviceAccount.rbac.enabled` creates a `ClusterRole` and
`ClusterRoleBinding`, allowing a ModelExpress server deployed in a dedicated
namespace to access metadata resources in a workload namespace:

```yaml
serviceAccount:
  rbac:
    enabled: true

env:
  MX_METADATA_BACKEND: kubernetes
```

## Upgrading

```bash
# Helm does not upgrade existing CRDs, so update them first.
kubectl apply -f helm/crds/modelexpress-crds.yaml

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

## Contributing

When contributing to this Helm chart, please ensure:

1. All templates follow Helm best practices
2. Values are properly documented
3. Examples are provided for common use cases
4. Tests are included for the chart
