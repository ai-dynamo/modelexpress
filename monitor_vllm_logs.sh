#!/bin/bash

# VLLM Decode Worker Log Monitor
# This script continuously monitors and displays logs from the VLLM decode worker pod

echo "🔍 VLLM Decode Worker Log Monitor"
echo "=================================="

# Function to get the current VLLM decode worker pod name
get_vllm_pod() {
    microk8s kubectl get po | grep vllmdecodeworker | awk '{print $1}' | head -1
}

# Function to check if pod is running
check_pod_status() {
    local pod_name=$1
    if [ -z "$pod_name" ]; then
        echo "❌ No VLLM decode worker pod found"
        return 1
    fi
    
    local status=$(microk8s kubectl get po "$pod_name" -o jsonpath='{.status.phase}' 2>/dev/null)
    if [ "$status" = "Running" ]; then
        return 0
    else
        echo "⏳ Pod $pod_name status: $status"
        return 1
    fi
}

# Function to display logs
show_logs() {
    local pod_name=$1
    echo "📋 Displaying logs for pod: $pod_name"
    echo "----------------------------------------"
    
    # Try to get logs from the pod
    if microk8s kubectl logs "$pod_name" 2>/dev/null; then
        echo "✅ Logs retrieved successfully"
    else
        echo "⚠️  Cannot retrieve logs from pod (may still be starting up)"
    fi
}

# Function to check persistent logs
check_persistent_logs() {
    local pod_name=$1
    echo "📁 Checking persistent logs in /model/logs/"
    echo "----------------------------------------"
    
    # Check if logs directory exists and show contents
    if microk8s kubectl exec "$pod_name" -- ls -la /model/logs/ 2>/dev/null; then
        echo ""
        echo "📄 Latest log files:"
        microk8s kubectl exec "$pod_name" -- find /model/logs/ -name "vllm_*.log" -type f -exec ls -la {} \; 2>/dev/null | tail -5
        
        echo ""
        echo "📖 Latest log content:"
        local latest_log=$(microk8s kubectl exec "$pod_name" -- find /model/logs/ -name "vllm_*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        if [ -n "$latest_log" ]; then
            echo "Showing content from: $latest_log"
            microk8s kubectl exec "$pod_name" -- tail -20 "$latest_log" 2>/dev/null
        else
            echo "No persistent log files found yet"
        fi
    else
        echo "⚠️  Cannot access persistent logs (pod may still be starting)"
    fi
}

# Main monitoring loop
echo "🚀 Starting log monitoring..."
echo "Press Ctrl+C to stop"
echo ""

while true; do
    # Get current pod name
    POD_NAME=$(get_vllm_pod)
    
    if [ -n "$POD_NAME" ]; then
        echo "🔍 Monitoring pod: $POD_NAME"
        
        # Check if pod is running
        if check_pod_status "$POD_NAME"; then
            echo "✅ Pod is running - checking logs..."
            
            # Show current logs
            show_logs "$POD_NAME"
            echo ""
            
            # Check persistent logs
            check_persistent_logs "$POD_NAME"
            echo ""
            
        else
            echo "⏳ Pod is not ready yet, waiting..."
        fi
    else
        echo "❌ No VLLM decode worker pod found, waiting..."
    fi
    
    echo "⏰ Waiting 10 seconds before next check..."
    echo "=========================================="
    sleep 10
done


