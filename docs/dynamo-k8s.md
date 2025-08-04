# Running Model Express with Dynamo on k8s

## 0. Prerequisites 

- [Install Dynamo Cloud on your Cluster](https://github.com/ai-dynamo/dynamo/blob/a8cb6554779f8283edd0c62d50743f2cb58e989b/docs/guides/dynamo_deploy/quickstart.md)

Should have respective pods running as shown here:

```
$ kubectl get po
NAME                                                              READY   STATUS    RESTARTS      AGE
dynamo-platform-dynamo-operator-controller-manager-54d48f4vdkh8   2/2     Running   6 (18h ago)   2d4h
dynamo-platform-etcd-0                                            1/1     Running   3 (18h ago)   2d4h
dynamo-platform-nats-0                                            2/2     Running   6 (18h ago)   2d4h
dynamo-platform-nats-box-5dbf45c748-vstcm                         1/1     Running   3 (18h ago)   2d4h
```


## 1. Use Model Express to Download a Model 

This section describes modelexpress workflow on a single node with both client and server. 

Ensure that the file structure here can be accessible for a volume we will mount into the cluster in a later step.

[Follow primary README.md for guidance on starting ModelExpress](https://github.com/ai-dynamo/modelexpress/blob/main/README.md)

#### Ensure Repository is cloned
```
$ git clone <repository-url>
$ cd ModelExpress
```

#### Build the project
```
$ cargo build
```

#### Start ModelExpress (will start on 0.0.0.0:8001 by default). 
```
$ cargo run --bin model_express_server

2025-08-01T22:22:50.568827Z  INFO model_express_server: Starting model_express_server with gRPC...
2025-08-01T22:22:50.568850Z  INFO model_express_server: Listening on gRPC endpoint: 0.0.0.0:8001
```

Open a seperate shell to use the CLI

#### Build Model Express CLI
```
$ cargo build --bin model-express-cli
```

Configure ModelExpress (can skip if using defaults)
```
$ ./target/release/model-express-cli model init

Enter your local cache mount path [~/.model-express/cache]: 
Enter your server endpoint [http://localhost:8001]: 
Auto-mount cache on startup? [Y/n]: 
Save this configuration? [Y/n]: 
ModelExpress Storage Configuration
===================================
Configuration saved successfully!
Storage path: "/home/kavink/.model-express/cache"
Server endpoint: http://localhost:8001
Auto-mount: true
```

#### Download a model using the CLI (should see more info on server log)
```
$ ./target/release/model-express-cli model download --endpoint http://localhost:8001 model download Qwen/Qwen3-0.6B

Model Download
  Model: Qwen/Qwen3-0.6B
  Provider: HuggingFace
  Strategy: SmartFallback


✅ SUCCESS
  Model 'Qwen/Qwen3-0.6B' downloaded successfully
```

### List the contents of Qwen in the Cache (for later step)
```
$ ls -al ~/.model-express/cache/models--Qwen--Qwen3-0.6B/snapshots/
total 12
drwxr-xr-x 3 kavink domain-users 4096 Aug  1 15:27 .
drwxr-xr-x 5 kavink domain-users 4096 Aug  1 15:27 ..
drwxr-xr-x 2 kavink domain-users 4096 Aug  1 15:27 c1899de289a04d12100db370d81485cdf75e47ca
```

Remember the hash path `/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca` for a later step when we start Dynamo

## 2. Create a K8s Persistent Volume and Claim for our Cache

Create a .yaml file with the following contents (eg. dynamo-models-volume.yaml)

Ensure spec.local.path has the path to your ModelExpress cache
```
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-express-cache-pv
  labels:
    type: local-storage
    app: model-express
spec:
  storageClassName: local-storage
  capacity:
    storage: 10Gi  # Adjust based on your model cache size requirements
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  local:
    path: /home/kavink/.model-express/cache
  nodeAffinity:
    required:
      nodeSelectorTerms:
      - matchExpressions:
        - key: kubernetes.io/hostname
          operator: In
          values:
          - your-node-name  # Replace with your actual node name
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-express-cache-pvc
  namespace: default  # Adjust namespace as needed
spec:
  storageClassName: local-storage
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi  # Should match the PV capacity
  selector:
    matchLabels:
      type: local-storage
      app: model-express 
```

Apply the .yaml config:
```
$ kubectl apply -f dynamo-models-volume.yaml
```

## 3. Deploy Dynamo on K8s with the Cache

Clone [Dynamo](https://github.com/ai-dynamo/dynamo):
```
$ git clone <repository-url>
$ cd dynamo
```

Open `./dynamo/components/backends/vllm/deploy/agg.yaml` and use the PVC to mount the cache to `/model` to both dynamo workers / services `Frontend` and `VllmDecodeWorker`:
```
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-agg
spec:
  services:
    Frontend:
      <..Other Contents Remain the Same..>
      pvc:
        name: model-express-cache-pvc
        mountPoint: /model
    VllmDecodeWorker:
      <..Other Contents Remain the Same..>
      pvc:
        name: model-express-cache-pvc
        mountPoint: /model
```

Additionally, we must specify the right path to the model hash snapshot (from above) relative to the `/model` \
```
VllmDecodeWorker:
    <..Other Contents Remain the Same..> 
          args:
            - "python3 -m dynamo.vllm --model /model/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca | tee /tmp/vllm.log"
    <..Other Contents Remain the Same..>
```

Lastly you will need to accquire an [access token from Huggingface](https://huggingface.co/docs/hub/en/security-tokens), and will need to create a kubernetes secret:
```
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=$HF_TOKEN
```

Now we can create the dynamo deployment:
```
kubectl apply -f ./dynamo/components/backends/vllm/deploy/agg.yaml

dynamographdeployment.nvidia.com/vllm-agg created
```

Should see two new pods, that will eventually load:
```
$ kubectl get po
NAME                                                              READY   STATUS    RESTARTS      AGE
dynamo-platform-dynamo-operator-controller-manager-54d48f4vdkh8   2/2     Running   6 (19h ago)   2d4h
dynamo-platform-etcd-0                                            1/1     Running   3 (19h ago)   2d4h
dynamo-platform-nats-0                                            2/2     Running   6 (19h ago)   2d4h
dynamo-platform-nats-box-5dbf45c748-vstcm                         1/1     Running   3 (19h ago)   2d4h
vllm-agg-frontend-fb896dffc-pnrfl                                 1/1     Running   0             38s
vllm-agg-vllmdecodeworker-67986d65b-7sx9j                         1/1     Running   0             38s
```


If you look at the logs of the decode worker, you can see we bypass downloading the model
```
$ kubectl logs vllm-agg-vllmdecodeworker-67986d65b-7sx9j 
```

Now lets make an inference request, by first port-forwarding the front end to our local 8000 port:
```
$ kubectl port-forward pods/vllm-agg-frontend-fb896dffc-pnrfl 8000:8000

Forwarding from [::1]:8000 -> 8000
```

Lets send the request now:
```
$ curl -s -N --no-buffer -X POST http://localhost:8000/v1/chat/completions -H 'accept: text/event-stream' -H 'Content-Type: application/json' -d '{"model": "Qwen/Qwen3-0.6B","messages": [{"role": "user","content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."}],"stream":false,"max_tokens": 30,"temperature": 0.0}'



{"id":"chatcmpl-b3234530-4da6-4319-8bc1-cd6d3e956214","choices":[{"index":0,"message":{"content":"<think>\nOkay, I need to develop a character background for the user's character in Eldoria. Let me start by understanding the requirements. The","refusal":null,"tool_calls":null,"role":"assistant","function_call":null,"audio":null},"finish_reason":"stop","logprobs":null}],"created":1754088562,"model":"Qwen/Qwen3-0.6B","service_tier":null,"system_fingerprint":null,"object":"chat.completion","usage":{"prompt_tokens":196,"completion_tokens":29,"total_tokens":225,"prompt_tokens_details":null,"completion_tokens_details":null}}
```

We have just demonstrated running dynamo with a cache managed by the ModelExpress



