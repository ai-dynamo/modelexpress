#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Stamp, lint, package and coherence-check the nightly helm chart.
#
# Called by nightly-ci.yml jobs package-helm-chart and stage-helm-ngc; keep
# both callers on this script so they cannot package different charts.
#
# Requires helm on PATH. Reads NIGHTLY_TAG, CHART_VERSION and NGC_STAGING_ORG
# from the environment; writes chart-dist/modelexpress-${CHART_VERSION}.tgz.

set -euo pipefail

: "${NIGHTLY_TAG:?NIGHTLY_TAG is required}"
: "${CHART_VERSION:?CHART_VERSION is required}"
: "${NGC_STAGING_ORG:?NGC_STAGING_ORG is required}"

NIGHTLY_REPO="nvcr.io/${NGC_STAGING_ORG}/ai-dynamo/modelexpress-server-nightly"

# values.yaml ships inside the packaged chart and points at the GA image, so
# it must be repointed at tonight's image. The values-*.yaml overlays are
# deploy-time inputs and do not ship in the chart.
python3 - "$NIGHTLY_REPO" "$NIGHTLY_TAG" <<'PY'
import re, sys
repo, tag = sys.argv[1], sys.argv[2]
p = "helm/values.yaml"
s = open(p).read()
s, nrepo = re.subn(r'(?m)^(\s*repository:\s*).*$', lambda m: m.group(1) + repo, s, count=1)
s, ntag = re.subn(r'(?m)^(\s*tag:\s*).*$', lambda m: m.group(1) + '"%s"' % tag, s, count=1)
if nrepo != 1 or ntag != 1:
    sys.exit("::error::failed to stamp image coordinates into helm/values.yaml "
             f"(repository matches={nrepo}, tag matches={ntag})")
open(p, "w").write(s)
PY
grep -A3 '^image:' helm/values.yaml

helm lint helm/
mkdir -p chart-dist
helm package helm/ \
  --version "${CHART_VERSION}" \
  --app-version "${NIGHTLY_TAG}" \
  --destination chart-dist/
ls -la chart-dist/

# Checked on the packaged tarball, which is what gets pushed. A chartmuseum
# push is immutable, so a chart pinned to a missing image cannot be corrected.
EXPECTED="${NIGHTLY_REPO}:${NIGHTLY_TAG}"
RENDERED=$(helm template mx "chart-dist/modelexpress-${CHART_VERSION}.tgz" | grep -E '^\s+image:' | head -1)
echo "rendered: ${RENDERED}"
case "${RENDERED}" in
  *"${EXPECTED}"*) echo "chart/image coherence OK" ;;
  *)
    echo "::error::packaged chart does not reference ${EXPECTED}"
    exit 1
    ;;
esac
