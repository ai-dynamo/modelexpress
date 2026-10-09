#!/bin/bash -e
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Description: This script triggers a GitLab mirror update and waits for completion
# Usage: ./update_gitlab_mirror.sh <gitlab_access_token> <gitlab_mirror_url>

GITLAB_ACCESS_TOKEN=$1
GITLAB_MIRROR_URL=$2

SYNC_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
curl --fail-with-body --connect-timeout 10 --max-time 30 \
    --request POST --header "PRIVATE-TOKEN: ${GITLAB_ACCESS_TOKEN}" "${GITLAB_MIRROR_URL}"

echo "Mirror update request submitted at ${SYNC_TIME}"

# Poll for completion
# GitLab limits the frequency of mirror updates to once every 5 mins.
# If you trigger an update more frequently, it may not start immediately.
# Make sure sync is finished after sync request was submitted
MAX_RETRIES=15
RETRY_INTERVAL=30
# Convert timestamps to epoch for comparison, timezone agnostic
SYNC_TIME_SECONDS=$(date -d "$SYNC_TIME" +%s)
for i in $(seq 1 $((MAX_RETRIES + 1))); do
    if MIRROR_INFO=$(curl --fail --silent --show-error \
        --connect-timeout 10 --max-time 30 --retry 3 --retry-delay 2 \
        --retry-connrefused --retry-max-time 120 \
        --header "PRIVATE-TOKEN: ${GITLAB_ACCESS_TOKEN}" "${GITLAB_MIRROR_URL}"); then
        MIRROR_STATUS=$(echo "$MIRROR_INFO" | jq -er '.update_status | select(type == "string")')
        LAST_UPDATE=$(echo "$MIRROR_INFO" | jq -r '.last_update_at // empty')
        LAST_ERROR=$(echo "$MIRROR_INFO" | jq -r '.last_error // empty')
        LAST_UPDATE_SECONDS=0
        if [ -n "$LAST_UPDATE" ]; then
            LAST_UPDATE_SECONDS=$(date -d "$LAST_UPDATE" +%s)
        fi
        echo "Mirror status: ${MIRROR_STATUS}; last update: ${LAST_UPDATE}; last error: ${LAST_ERROR}"
        if [ "$LAST_UPDATE_SECONDS" -gt "$SYNC_TIME_SECONDS" ]; then
            case "$MIRROR_STATUS" in
                finished)
                    echo "Mirror sync successful. Last update: $LAST_UPDATE"
                    exit 0
                    ;;
                failed|canceled)
                    echo "Mirror sync failed: $LAST_ERROR"
                    exit 1
                    ;;
            esac
        fi
    else
        status=$?
        case "$status" in
            5|6|7|18|28|35|52|55|56)
                echo "Mirror status request failed with curl exit ${status}; retrying"
                ;;
            *) exit "$status" ;;
        esac
    fi
    if [ "$i" -gt "$MAX_RETRIES" ]; then
        break
    fi
    echo "Waiting for mirror sync to complete. Attempt $i of $MAX_RETRIES"
    sleep $RETRY_INTERVAL
done
echo "Mirror sync failed or timed out"
exit 1
