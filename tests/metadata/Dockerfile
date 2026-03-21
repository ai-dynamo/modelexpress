FROM python:3.12-slim

RUN pip install --no-cache-dir cryptography "protobuf>=5.27.0,<6.0.0" grpcio

WORKDIR /app

# Install mx_libp2p (DHT library)
COPY mx_libp2p/ mx_libp2p_pkg/
RUN pip install --no-cache-dir ./mx_libp2p_pkg/

# Install only the files we need from modelexpress - NOT the full package.
# The full package's __init__.py imports gds_loader -> torch, which we don't
# need and don't want to install. Copy just the proto stubs, dht_client, and
# a minimal __init__.py.
RUN mkdir -p /usr/local/lib/python3.12/site-packages/modelexpress
COPY modelexpress_client/python/modelexpress/p2p_pb2.py \
     modelexpress_client/python/modelexpress/p2p_pb2_grpc.py \
     modelexpress_client/python/modelexpress/dht_client.py \
     modelexpress_client/python/modelexpress/client.py \
     /usr/local/lib/python3.12/site-packages/modelexpress/
RUN echo '# Minimal init for metadata test' > /usr/local/lib/python3.12/site-packages/modelexpress/__init__.py

COPY tests/metadata/k8s_metadata_test.py .

ENTRYPOINT ["python", "k8s_metadata_test.py"]
