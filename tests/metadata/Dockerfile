FROM python:3.12-slim

RUN pip install --no-cache-dir cryptography "protobuf>=5.27.0,<6.0.0" grpcio

WORKDIR /app

# Install mx_libp2p (DHT library)
COPY mx_libp2p/ mx_libp2p_pkg/
RUN pip install --no-cache-dir ./mx_libp2p_pkg/

# Install modelexpress client (DhtMetadataClient, protobuf types)
# Use --no-deps to avoid pulling torch, nixl, and other heavy runtime deps.
# This test only needs the protobuf types and DhtMetadataClient.
COPY modelexpress_client/python/ mx_client_pkg/
RUN pip install --no-cache-dir --no-deps ./mx_client_pkg/

COPY tests/metadata/k8s_metadata_test.py .

ENTRYPOINT ["python", "k8s_metadata_test.py"]
