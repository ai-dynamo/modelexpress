#!/usr/bin/env bash
# Source only from the scripts below, with the rendered run directory as $1.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${1:?usage: script RESULTS_DIR}/run.env"
: "${WORKER_CONTEXT:?render an explicit Kubernetes context}"
: "${NAMESPACE:?render an explicit namespace}"
k() { kubectl --context "$WORKER_CONTEXT" -n "$NAMESPACE" "$@"; }
CONTROL_POD="$RESOURCE_PREFIX-control"
SOURCE_POD="$RESOURCE_PREFIX-s3"
PEER_POD="$RESOURCE_PREFIX-peer"
