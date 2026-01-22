#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ModelExpress Helm Chart Deployment Script
# This script helps deploy ModelExpress using Helm

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
RELEASE_NAME="modelexpress"
NAMESPACE="modelexpress"
VALUES_FILE=""
DRY_RUN=false
UPGRADE=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -r, --release-name NAME    Release name (default: modelexpress)
    -n, --namespace NAME       Kubernetes namespace (default: modelexpress)
    -f, --values FILE          Values file to use (e.g., values-production.yaml)
    -d, --dry-run              Perform a dry run
    -u, --upgrade              Upgrade existing release
    -h, --help                 Show this help message

Available values files (in script directory):
    values-production.yaml     Production configuration
    values-development.yaml    Development configuration  
    values.yaml               Default configuration
    test-values.yaml          Test configuration

Examples:
    # Deploy with default values
    $0

    # Deploy with production values
    $0 -f values-production.yaml

    # Deploy with custom release name and namespace
    $0 -r my-modelexpress -n my-namespace

    # Perform a dry run
    $0 -d

    # Upgrade existing release
    $0 -u -f values-production.yaml

EOF
}

# Function to find default values files
find_default_values() {
    local script_dir=$(dirname "$0")
    local available_files=()
    
    # Check for common values files in the script directory
    for file in "values-production.yaml" "values-development.yaml" "values.yaml" "test-values.yaml"; do
        if [[ -f "$script_dir/$file" ]]; then
            available_files+=("$file")
        fi
    done
    
    if [[ ${#available_files[@]} -gt 0 ]]; then
        print_status "Available values files in $(basename "$script_dir"):"
        for i in "${!available_files[@]}"; do
            echo "  $((i+1)). ${available_files[$i]}"
        done
        echo ""
    fi
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check for kubectl or microk8s kubectl (prioritize microk8s if available)
    KUBECTL_CMD=""
    if command -v microk8s &> /dev/null && microk8s kubectl version --client &> /dev/null; then
        KUBECTL_CMD="microk8s kubectl"
        # Set KUBECONFIG for MicroK8s
        export KUBECONFIG=/var/snap/microk8s/current/credentials/client.config
        print_status "Using microk8s kubectl (KUBECONFIG set to MicroK8s)"
    elif command -v kubectl &> /dev/null; then
        KUBECTL_CMD="kubectl"
        print_status "Using kubectl"
    else
        print_error "Neither kubectl nor microk8s kubectl is available. Please install one of them."
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        print_error "Helm is not installed. Please install Helm first."
        exit 1
    fi
    
    # Check if we can connect to Kubernetes cluster
    if ! $KUBECTL_CMD cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    # Check if we can access Docker Hub (optional check)
    if ! curl -s --max-time 5 https://registry-1.docker.io/v2/ > /dev/null; then
        print_warning "Cannot access Docker Hub. Ensure you have internet connectivity."
    fi
    
    print_success "Prerequisites check passed"
}

# Function to create namespace if it doesn't exist
create_namespace() {
    if ! $KUBECTL_CMD get namespace "$NAMESPACE" &> /dev/null; then
        print_status "Creating namespace: $NAMESPACE"
        $KUBECTL_CMD create namespace "$NAMESPACE"
        print_success "Namespace created: $NAMESPACE"
    else
        print_status "Namespace already exists: $NAMESPACE"
    fi
}

# Function to deploy the chart
deploy_chart() {
    local helm_args=()
    
    if [ "$DRY_RUN" = true ]; then
        helm_args+=("--dry-run")
        print_status "Performing dry run..."
    fi
    
    if [ -n "$VALUES_FILE" ]; then
        helm_args+=("-f" "$VALUES_FILE")
    fi
    
    helm_args+=("--namespace" "$NAMESPACE")
    
    if [ "$UPGRADE" = true ]; then
        print_status "Upgrading release: $RELEASE_NAME"
        helm upgrade "${helm_args[@]}" "$RELEASE_NAME" .
    else
        print_status "Installing release: $RELEASE_NAME"
        helm install "${helm_args[@]}" "$RELEASE_NAME" .
    fi
    
    print_success "Deployment completed successfully"
}

# Function to show deployment status
show_status() {
    print_status "Checking deployment status..."
    
    echo
    echo "Pods:"
    $KUBECTL_CMD get pods -n "$NAMESPACE" -l app.kubernetes.io/name=modelexpress
    
    echo
    echo "Services:"
    $KUBECTL_CMD get svc -n "$NAMESPACE" -l app.kubernetes.io/name=modelexpress
    
    echo
    echo "PersistentVolumeClaims:"
    $KUBECTL_CMD get pvc -n "$NAMESPACE" -l app.kubernetes.io/name=modelexpress
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--release-name)
            RELEASE_NAME="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -f|--values)
            VALUES_FILE="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -u|--upgrade)
            UPGRADE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_status "Starting ModelExpress Helm deployment..."
    print_status "Release: $RELEASE_NAME"
    print_status "Namespace: $NAMESPACE"
    if [[ -n "$VALUES_FILE" ]]; then
        print_status "Values file: $VALUES_FILE"
    else
        print_status "Values file: Using default values"
    fi
    
    find_default_values
    check_prerequisites
    create_namespace
    deploy_chart
    
    if [ "$DRY_RUN" = false ]; then
        show_status
        print_success "Deployment completed!"
        print_status "To access the service, run: $KUBECTL_CMD port-forward -n $NAMESPACE svc/$RELEASE_NAME 8001:8001"
    fi
}

# Run main function
main
