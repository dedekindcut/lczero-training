#!/bin/bash

################################################################################
# LCZero Training LoRA Setup Script
#
# This script sets up the environment for LCZero training with LoRA support.
# It handles:
# - System dependencies installation
# - Python environment setup with uv
# - Repository cloning and configuration
# - Network file downloads
# - Dataset downloads
# - Environment variable configuration
#
# Usage:
#   ./startup.sh [OPTIONS]
#
# Options:
#   --skip-deps       Skip system dependency installation
#   --skip-networks   Skip network file downloads
#   --skip-data       Skip dataset downloads
#   --run-training    Run training after setup (default: false)
#   --workspace DIR   Set workspace directory (default: /workspace)
#   -h, --help        Show this help message
#
################################################################################

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# ============================================================================
# Configuration
# ============================================================================

WORKSPACE="${WORKSPACE:-/workspace}"
SKIP_DEPS=false
SKIP_NETWORKS=false
SKIP_DATA=false
RUN_TRAINING=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    sed -n '/^# Usage:/,/^################################################################################$/p' "$0" | \
        grep -v "^#####" | sed 's/^# \?//'
    exit 0
}

check_command() {
    if command -v "$1" &> /dev/null; then
        log_success "$1 is available"
        return 0
    else
        log_warning "$1 is not available"
        return 1
    fi
}

cleanup_on_error() {
    log_error "Script failed. Check the error messages above."
    exit 1
}

# Set up error trap
trap cleanup_on_error ERR

# ============================================================================
# Parse Command Line Arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --skip-networks)
            SKIP_NETWORKS=true
            shift
            ;;
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --run-training)
            RUN_TRAINING=true
            shift
            ;;
        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# ============================================================================
# Main Setup Process
# ============================================================================

log_info "Starting LCZero Training LoRA setup..."
log_info "Workspace directory: $WORKSPACE"

# Create workspace if it doesn't exist
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

# ----------------------------------------------------------------------------
# 1. Install System Dependencies
# ----------------------------------------------------------------------------

if [ "$SKIP_DEPS" = false ]; then
    log_info "Installing system dependencies..."

    if [ "$(id -u)" -ne 0 ]; then
        log_error "System dependency installation requires root privileges"
        log_warning "Run with sudo or use --skip-deps if dependencies are already installed"
        exit 1
    fi

    apt-get update
    apt-get install -y \
        git \
        build-essential \
        python3-dev \
        protobuf-compiler \
        rsync \
        curl \
        wget \
        unzip

    log_success "System dependencies installed"
else
    log_info "Skipping system dependency installation"
fi

# ----------------------------------------------------------------------------
# 2. Install uv (Python package manager)
# ----------------------------------------------------------------------------

if ! check_command uv; then
    log_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Source the environment file
    if [ -f "$HOME/.local/bin/env" ]; then
        source "$HOME/.local/bin/env"
    fi

    # Add to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"

    log_success "uv installed"
else
    log_info "uv already installed"
fi

# ----------------------------------------------------------------------------
# 3. Install croc (file transfer tool)
# ----------------------------------------------------------------------------

if ! check_command croc; then
    log_info "Installing croc..."
    curl -s https://getcroc.schollz.com | bash
    log_success "croc installed"
else
    log_info "croc already installed"
fi

# ----------------------------------------------------------------------------
# 4. Clone LCZero Training Repository
# ----------------------------------------------------------------------------

LCZERO_DIR="$WORKSPACE/lczero-training"

if [ -d "$LCZERO_DIR" ]; then
    log_warning "LCZero training directory already exists at $LCZERO_DIR"
    log_info "Pulling latest changes..."
    cd "$LCZERO_DIR"
    git pull || log_warning "Failed to pull latest changes (might be due to local modifications)"
else
    log_info "Cloning LCZero training repository..."
    git clone https://github.com/dedekindcut/lczero-training -b lora --single-branch "$LCZERO_DIR"
    log_success "Repository cloned"
fi

cd "$LCZERO_DIR"

# ----------------------------------------------------------------------------
# 5. Set Up Python Environment
# ----------------------------------------------------------------------------

log_info "Setting up Python environment with uv..."

# Initialize uv project if not already done
if [ ! -f "pyproject.toml" ]; then
    uv init
fi

# Install Python dependencies
log_info "Installing Python packages (this may take a while)..."
uv add "tensorflow[and-cuda]" tf_keras pyyaml "protobuf<5.0" tensorflow-addons numpy

log_success "Python environment configured"

# ----------------------------------------------------------------------------
# 6. Compile Protocol Buffers
# ----------------------------------------------------------------------------

log_info "Compiling protocol buffers..."

mkdir -p proto
touch proto/__init__.py

if [ -f "tf/net.proto" ]; then
    protoc -I=tf --python_out=proto tf/net.proto
    log_success "Protocol buffers compiled"
else
    log_error "tf/net.proto not found"
    exit 1
fi

# ----------------------------------------------------------------------------
# 7. Download Network Files
# ----------------------------------------------------------------------------

cd "$WORKSPACE"
mkdir -p nets

if [ "$SKIP_NETWORKS" = false ]; then
    log_info "Downloading network files..."

    cd nets

    NETWORK1="BT4-1024x15x32h-swa-6147500.pb.gz"
    NETWORK2="t3-512x15x16h-distill-swa-2767500.pb.gz"

    if [ ! -f "$NETWORK1" ]; then
        log_info "Downloading $NETWORK1..."
        wget -q --show-progress https://storage.lczero.org/files/networks-contrib/big-transformers/"$NETWORK1"
    else
        log_info "$NETWORK1 already exists"
    fi

    if [ ! -f "$NETWORK2" ]; then
        log_info "Downloading $NETWORK2..."
        wget -q --show-progress https://storage.lczero.org/files/networks-contrib/t3-512x15x16h-distill-swa-2767500.pb.gz
    else
        log_info "$NETWORK2 already exists"
    fi

    log_success "Network files downloaded"
else
    log_info "Skipping network file downloads"
fi

# ----------------------------------------------------------------------------
# 8. Download Datasets
# ----------------------------------------------------------------------------

cd "$WORKSPACE"
mkdir -p data

if [ "$SKIP_DATA" = false ]; then
    log_info "Downloading datasets..."

    cd data

    # Check for HuggingFace token
    if [ -z "${HF_TOKEN:-}" ]; then
        log_error "HF_TOKEN environment variable not set"
        log_error "Please set HF_TOKEN before running: export HF_TOKEN='your_token_here'"
        exit 1    fi

    DATASET_FILE="RookOddsV8.zip"
    if [ ! -f "$DATASET_FILE" ]; then
        log_info "Downloading $DATASET_FILE from HuggingFace..."
        wget -q --show-progress \
            --header="Authorization: Bearer $HF_TOKEN" \
            "https://huggingface.co/datasets/Naca1208/RookOddsV8/resolve/main/RookOddsV8.zip"
        log_success "$DATASET_FILE downloaded"
    else
        log_info "$DATASET_FILE already exists"
    fi

    # Install gdown if needed
    if ! check_command gdown; then
        log_info "Installing gdown..."
        uv tool install gdown
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Download from Google Drive
    GDRIVE_FILE_ID="1BZlyhEFwgu_wWV251RKFWjCY7WAuzqdG"
    log_info "Downloading file from Google Drive..."
    gdown "$GDRIVE_FILE_ID" || log_warning "Failed to download from Google Drive"

    log_success "Datasets downloaded"
else
    log_info "Skipping dataset downloads"
fi

# ----------------------------------------------------------------------------
# 9. Configure Environment Variables
# ----------------------------------------------------------------------------

log_info "Configuring environment variables..."

# Create environment file
ENV_FILE="$LCZERO_DIR/.env"
cat > "$ENV_FILE" << 'EOF'
export TF_USE_LEGACY_KERAS=1
export PYTHONPATH=.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
EOF

log_success "Environment variables configured in $ENV_FILE"
log_info "Source this file before running training: source $ENV_FILE"

# Source for current session
source "$ENV_FILE"

# ============================================================================
# Completion
# ============================================================================

echo ""
log_success "=========================================="
log_success "Setup completed successfully!"
log_success "=========================================="
echo ""
log_info "Workspace: $WORKSPACE"
log_info "LCZero directory: $LCZERO_DIR"
log_info "Networks: $WORKSPACE/nets"
log_info "Data: $WORKSPACE/data"
echo ""

# ----------------------------------------------------------------------------
# 10. Optional: Run Training
# ----------------------------------------------------------------------------

if [ "$RUN_TRAINING" = true ]; then
    log_info "Starting training..."
    cd "$LCZERO_DIR"

    if [ -f "tf/configs/example.yaml" ]; then
        python3 tf/train.py --cfg tf/configs/example.yaml
    else
        log_error "Training config not found at tf/configs/example.yaml"
        exit 1
    fi
else
    log_info "To run training manually:"
    echo ""
    echo "  cd $LCZERO_DIR"
    echo "  source .env"
    echo "  python3 tf/train.py --cfg tf/configs/example.yaml"
    echo ""
fi
