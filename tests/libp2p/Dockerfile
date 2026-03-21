FROM python:3.12-slim

RUN pip install --no-cache-dir cryptography "protobuf>=5.27.0,<6.0.0"

# kubectl for soak test (rollout restart)
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -sLo /usr/local/bin/kubectl \
       "https://dl.k8s.io/release/$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && chmod +x /usr/local/bin/kubectl \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY mx_libp2p/ mx_libp2p_pkg/
RUN pip install --no-cache-dir ./mx_libp2p_pkg/

COPY tests/libp2p/k8s_dht_runner.py .

ENTRYPOINT ["python", "k8s_dht_runner.py"]
