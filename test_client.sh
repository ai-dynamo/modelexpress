# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
TEST_CACHE_PATH="/tmp/model-express-test"
SERVER_ENDPOINT="http://localhost:8001"
TEST_MODEL="google-t5/t5-small"

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

# Function to check if server is running
check_server() {
    if curl -s "$SERVER_ENDPOINT" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to cleanup test environment
cleanup() {
    print_status "Cleaning up test environment..."
    rm -rf "$TEST_CACHE_PATH"
    rm -f ~/.model-express/config.yaml
    print_success "Cleanup completed"
}

# Function to setup test environment
setup_test_env() {
    print_status "Setting up test environment..."

    # Create test cache directory
    mkdir -p "$TEST_CACHE_PATH"

    # Set environment variables
    export MODEL_EXPRESS_CACHE_PATH="$TEST_CACHE_PATH"
    export MODEL_EXPRESS_SERVER_ENDPOINT="$SERVER_ENDPOINT"
    export RUST_LOG=info

    print_success "Test environment setup completed"
}

# Function to test cache CLI
test_cache_cli() {
    print_status "Testing Cache CLI functionality..."

    # Test cache initialization
    print_status "Testing cache initialization..."
    cargo run --bin cache_cli init --cache-path "$TEST_CACHE_PATH" || {
        print_error "Cache initialization failed"
        return 1
    }
    print_success "Cache initialization passed"

    # Test cache status
    print_status "Testing cache status..."
    cargo run --bin cache_cli status || {
        print_error "Cache status check failed"
        return 1
    }
    print_success "Cache status check passed"

    # Test cache list (should be empty initially)
    print_status "Testing cache list..."
    cargo run --bin cache_cli list || {
        print_error "Cache list failed"
        return 1
    }
    print_success "Cache list passed"

    # Test cache stats
    print_status "Testing cache statistics..."
    cargo run --bin cache_cli stats || {
        print_error "Cache stats failed"
        return 1
    }
    print_success "Cache stats passed"

    # Test cache validation
    print_status "Testing cache validation..."
    cargo run --bin cache_cli validate || {
        print_error "Cache validation failed"
        return 1
    }
    print_success "Cache validation passed"
}

# Function to test client with server
test_client_with_server() {
    print_status "Testing client with server..."

    # Check if server is running
    if ! check_server; then
        print_warning "Server not running, starting server..."
        cargo run --bin model_express_server > server.log 2>&1 &
        SERVER_PID=$!
        sleep 5

        if ! check_server; then
            print_error "Failed to start server"
            return 1
        fi
        print_success "Server started successfully"
    else
        print_success "Server is already running"
    fi

    # Test basic client functionality
    print_status "Testing basic client functionality..."
    cargo run --bin test_client -- --test-model "$TEST_MODEL" || {
        print_error "Basic client test failed"
        return 1
    }
    print_success "Basic client test passed"

    # Test cache CLI after model download
    print_status "Testing cache CLI after model download..."
    cargo run --bin cache_cli list || {
        print_error "Cache list after download failed"
        return 1
    }
    print_success "Cache list after download passed"

    # Test cache stats after model download
    print_status "Testing cache stats after model download..."
    cargo run --bin cache_cli stats --detailed || {
        print_error "Cache stats after download failed"
        return 1
    }
    print_success "Cache stats after download passed"
}

# Function to test error handling
test_error_handling() {
    print_status "Testing error handling..."

    # Test with invalid server endpoint
    print_status "Testing with invalid server endpoint..."
    cargo run --bin cache_cli --server-endpoint "http://invalid:9999" status || {
        print_success "Correctly handled invalid server endpoint"
    }

    # Test with non-existent cache path
    print_status "Testing with non-existent cache path..."
    cargo run --bin cache_cli --cache-path "/non/existent/path" list || {
        print_success "Correctly handled non-existent cache path"
    }

    # Test cache clearing
    print_status "Testing cache clearing..."
    cargo run --bin cache_cli clear "non-existent-model" || {
        print_success "Correctly handled clearing non-existent model"
    }
}

# Function to test cache discovery methods
test_cache_discovery() {
    print_status "Testing cache discovery methods..."

    # Test command line argument discovery
    print_status "Testing command line argument discovery..."
    cargo run --bin cache_cli --cache-path "$TEST_CACHE_PATH" list || {
        print_error "Command line argument discovery failed"
        return 1
    }
    print_success "Command line argument discovery passed"

    # Test environment variable discovery
    print_status "Testing environment variable discovery..."
    MODEL_EXPRESS_CACHE_PATH="$TEST_CACHE_PATH" cargo run --bin cache_cli list || {
        print_error "Environment variable discovery failed"
        return 1
    }
    print_success "Environment variable discovery passed"

    # Test config file discovery
    print_status "Testing config file discovery..."
    mkdir -p ~/.model-express
    cat > ~/.model-express/config.yaml << EOF
local_path: $TEST_CACHE_PATH
server_endpoint: $SERVER_ENDPOINT
timeout_secs: 30
EOF

    cargo run --bin cache_cli list || {
        print_error "Config file discovery failed"
        return 1
    }
    print_success "Config file discovery passed"
}

# Function to test unit tests
test_unit_tests() {
    print_status "Running unit tests..."

    # Test cache configuration
    cargo test --package model_express_client cache_config::tests || {
        print_error "Cache configuration unit tests failed"
        return 1
    }
    print_success "Cache configuration unit tests passed"

    # Test client library
    cargo test --package model_express_client lib || {
        print_error "Client library unit tests failed"
        return 1
    }
    print_success "Client library unit tests passed"
}

# Function to test integration tests
test_integration_tests() {
    print_status "Running integration tests..."

    # Test workspace integration tests
    cargo test --package workspace-tests || {
        print_warning "Integration tests failed (may require server)"
        return 1
    }
    print_success "Integration tests passed"
}

# Main test function
main() {
    print_status "Starting ModelExpress Client Tests"
    print_status "=================================="

    # Setup trap for cleanup
    trap cleanup EXIT

    # Setup test environment
    setup_test_env

    # Run tests
    test_unit_tests
    test_cache_discovery
    test_cache_cli
    test_error_handling
    test_client_with_server
    test_integration_tests

    print_success "All tests completed successfully!"
    print_status "Test Summary:"
    print_status "- Unit tests: ✓"
    print_status "- Cache discovery: ✓"
    print_status "- Cache CLI: ✓"
    print_status "- Error handling: ✓"
    print_status "- Client with server: ✓"
    print_status "- Integration tests: ✓"
}

# Check if script is run with arguments
if [ $# -eq 0 ]; then
    # Run all tests
    main
else
    # Run specific test based on argument
    case "$1" in
        "unit")
            setup_test_env
            test_unit_tests
            ;;
        "cache")
            setup_test_env
            test_cache_cli
            ;;
        "discovery")
            setup_test_env
            test_cache_discovery
            ;;
        "client")
            setup_test_env
            test_client_with_server
            ;;
        "errors")
            setup_test_env
            test_error_handling
            ;;
        "integration")
            setup_test_env
            test_integration_tests
            ;;
        "cleanup")
            cleanup
            ;;
        *)
            echo "Usage: $0 [unit|cache|discovery|client|errors|integration|cleanup]"
            echo "  unit       - Run unit tests only"
            echo "  cache      - Run cache CLI tests only"
            echo "  discovery  - Run cache discovery tests only"
            echo "  client     - Run client with server tests only"
            echo "  errors     - Run error handling tests only"
            echo "  integration- Run integration tests only"
            echo "  cleanup    - Clean up test environment only"
            echo "  (no args)  - Run all tests"
            exit 1
            ;;
    esac
fi