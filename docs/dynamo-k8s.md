# Running Model Express with Dynamo on k8s

### Prerequisites 

- [Install Dynamo Cloud on your Cluster](https://github.com/ai-dynamo/dynamo/blob/a8cb6554779f8283edd0c62d50743f2cb58e989b/docs/guides/dynamo_deploy/dynamo_cloud.md)

Should have respective pods running as shown here:

```
$ kubectl get po
NAME                                                              READY   STATUS    RESTARTS      AGE
dynamo-platform-dynamo-operator-controller-manager-54d48f4vdkh8   2/2     Running   6 (18h ago)   2d4h
dynamo-platform-etcd-0                                            1/1     Running   3 (18h ago)   2d4h
dynamo-platform-nats-0                                            2/2     Running   6 (18h ago)   2d4h
dynamo-platform-nats-box-5dbf45c748-vstcm                         1/1     Running   3 (18h ago)   2d4h
kavink@nnoble-desktop:~/model-express/docs$ 
```


## 1. Use Model Express to Download a Model

[Follow primary README.md for guidance on starting ModelExpress](https://github.com/ai-dynamo/modelexpress/blob/main/README.md)

```
## Ensure Repository is cloned ##

$ git clone <repository-url>
$ cd ModelExpress

## Build the project ##

$ cargo build

## Start ModelExpress (will start on 0.0.0.0:8001 by default). ##

$ cargo run --bin model_express_server
```

In a seperate shell, lets use the CLI

```
## Build Model Express CLI ##

$ cargo build --bin model-express-cli

## Configure ModelExpress (can skip if using defaults) ##

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


