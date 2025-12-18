# Lc0 Training Setup on Vast.ai with `uv`

This guide documents the complete process for setting up a fresh Vast.ai GPU instance and running the Leela Chess Zero (Lc0) training pipeline using `uv`.

## 1. Instance Selection (Vast.ai)

*   **Image:** Use a CUDA-enabled Docker image. Recommended: `nvidia/cuda:12.2.0-devel-ubuntu22.04` (or similar 11.8+/12.x).
*   **GPU:** RTX 4090 (24GB VRAM) is highly recommended for cost/performance. A100/H100 is faster but more expensive.
*   **Disk Space:** Ensure you allocate enough disk space (e.g., 64GB+) for the dataset and checkpoints.

## 2. Initial Connection & Setup

SSH into your instance using the command provided by Vast.ai:
```bash
ssh -p <PORT> root@<IP>
```

### Install System Dependencies
Run these commands on the remote instance to install Python build tools, git, and protobuf compiler:

```bash
apt-get update
apt-get install -y git build-essential python3-dev protobuf-compiler rsync curl
```

### Install `uv` (Modern Python Tool)
Install `uv` for fast dependency management:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

## 3. Project Setup

Clone your repository or sync your local project to the instance.

**Option A: Git Clone**
```bash
git clone <your-repo-url> lczero-training
cd lczero-training
```

**Option B: RSYNC from Local Machine**
From your local terminal:
```bash
rsync -avz -e "ssh -p <PORT>" ./ root@<IP>:~/lczero-training/
```

## 4. Python Environment & Dependencies

Initialize the project and install dependencies using `uv`. This handles the complex TensorFlow/CUDA version matching automatically.

```bash
cd ~/lczero-training
uv init
```

**Install Core Libraries:**
We use `tensorflow[and-cuda]` to bundle the correct NVIDIA libraries, avoiding system-level driver conflicts.

```bash
uv add "tensorflow[and-cuda]" tf_keras pyyaml "protobuf<5.0" tensorflow-addons numpy
```

## 5. Compile Protobufs

Compile the network definition protocol buffers.

```bash
mkdir -p proto
touch proto/__init__.py
protoc -I=tf --python_out=proto tf/net.proto
```

## 6. Data & Model Setup

1.  **Dataset:** Ensure your training chunks (`.gz` files) are in a known directory (e.g., `/workspace/data/`).
2.  **Base Model:** If fine-tuning (LoRA), place your `.pb.gz` base weights file in `/workspace/nets/`.
3.  **Configuration:** Edit `tf/configs/example.yaml`.
    *   **Critical:** Match `input_type`, `embedding_style`, and `embedding_dense_sz` to your base model.
    *   **Vast.ai Paths:** Update `input_train` and `pb_source` to point to the correct locations on the instance.
    *   **Batch Size:** For RTX 4090, start with `batch_size: 256` or `512`. If OOM, try `128`.

## 7. Launching Training

Use the following command to start training. It sets necessary environment variables for legacy Keras compatibility.

```bash
source .venv/bin/activate
export TF_USE_LEGACY_KERAS=1
export PYTHONPATH=.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python3 tf/train.py --cfg tf/configs/example.yaml
```

## 8. Converting the Model

Use the following command to convert the checkpoint to `.pb.gz`.

```bash
python3 tf/model_to_net.py --cfg tf/configs/example.yaml
```


**Monitoring:**
*   **Console:** Watch for `P Acc` (Policy Accuracy) and speed (`pos/s`).
*   **TensorBoard:**
    On Remote: `uv run tensorboard --logdir leelalogs --port 6006`
    On Local: `ssh -L 6006:localhost:6006 root@<IP> -p <PORT>`
    Browser: `http://localhost:6006`

## Troubleshooting Common Issues

*   **OOM (ResourceExhaustedError):**
    *   Check for phantom processes: `fuser -v /dev/nvidia0` -> `kill -9 <PID>`.
    *   Reduce `batch_size` in `example.yaml`.
    *   Reduce `shuffle_size`.

*   **"Blind" Model (Low Accuracy / Random Moves):**
    *   The model architecture in `example.yaml` does not match the weights file.
    *   Check `encoder_rms_norm`, `omit_qkv_biases`, and `embedding_style`. Use `inspect_pb_gz.py` (if available) or check the original model config.

*   **NaN Loss:**
    *   Learning Rate is too high. Reduce `lr_values` (e.g., `0.0001`).
