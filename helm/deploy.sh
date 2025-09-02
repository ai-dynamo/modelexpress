#!/bin/bash

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
RELEASE_NAME="model-express"
NAMESPACE="model-express"
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
    -r, --release-name NAME    Release name (default: model-express)
    -n, --namespace NAME       Kubernetes namespace (default: model-express)
    -f, --values FILE          Values file to use (e.g., values-production.yaml)
    -d, --dry-run              Perform a dry run
    -u, --upgrade              Upgrade existing release
    -h, --help                 Show this help message

Examples:
    # Deploy with default values
    $0

    # Deploy with production values
    $0 -f values-production.yaml

    # Deploy with custom release name and namespace
    $0 -r my-model-express -n my-namespace

    # Perform a dry run
    $0 -d

    # Upgrade existing release
    $0 -u -f values-production.yaml

EOF
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        print_error "Helm is not installed. Please install Helm first."
        exit 1
    fi
    
    # Check if we can connect to Kubernetes cluster
    if ! kubectl cluster-info &> /dev/null; then
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
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        print_status "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
        print_success "Namespace created: $NAMESPACE"
    else
        print_status "Namespace already exists: $NAMESPACE"
    fi
}

# Function to deploy the chart
deploy_chart() {
    local helm_cmd="helm"
    
    if [ "$DRY_RUN" = true ]; then
        helm_cmd="$helm_cmd --dry-run"
        print_status "Performing dry run..."
    fi
    
    if [ "$UPGRADE" = true ]; then
        print_status "Upgrading release: $RELEASE_NAME"
        $helm_cmd upgrade "$RELEASE_NAME" . --namespace "$NAMESPACE"
    else
        print_status "Installing release: $RELEASE_NAME"
        $helm_cmd install "$RELEASE_NAME" . --namespace "$NAMESPACE"
    fi
    
    if [ -n "$VALUES_FILE" ]; then
        helm_cmd="$helm_cmd -f $VALUES_FILE"
    fi
    
    print_success "Deployment completed successfully"
}

# Function to show deployment status
show_status() {
    print_status "Checking deployment status..."
    
    echo
    echo "Pods:"
    kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=model-express
    
    echo
    echo "Services:"
    kubectl get svc -n "$NAMESPACE" -l app.kubernetes.io/name=model-express
    
    echo
    echo "PersistentVolumeClaims:"
    kubectl get pvc -n "$NAMESPACE" -l app.kubernetes.io/name=model-express
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
    
    check_prerequisites
    create_namespace
    deploy_chart
    
    if [ "$DRY_RUN" = false ]; then
        show_status
        print_success "Deployment completed!"
        print_status "To access the service, run: kubectl port-forward -n $NAMESPACE svc/$RELEASE_NAME 8001:8001"
    fi
}

# Run main function
main
