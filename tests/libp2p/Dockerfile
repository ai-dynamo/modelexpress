FROM python:3.12-slim

RUN pip install --no-cache-dir cryptography protobuf

WORKDIR /app
COPY mx_libp2p/ mx_libp2p_pkg/
RUN pip install --no-cache-dir ./mx_libp2p_pkg/

COPY tests/libp2p/k8s_dht_runner.py .

ENTRYPOINT ["python", "k8s_dht_runner.py"]
