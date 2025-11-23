# Lc0 Training Setup with `uv`

This guide documents the process for setting up and running the Leela Chess Zero (Lc0) training pipeline using `uv` for modern, fast Python dependency management. It addresses specific compatibility issues with newer TensorFlow versions and legacy Lc0 codebases.

## Prerequisites

*   **OS:** Linux (Ubuntu recommended) or macOS.
*   **Hardware:** NVIDIA GPU with recent drivers (CUDA 11.8+ / 12.x compatible).
*   **Tools:** `uv` installed ([Installation Guide](https://github.com/astral-sh/uv)).
*   **System Packages:** `protobuf-compiler` (for `protoc`).
    ```bash
    # Ubuntu/Debian
    sudo apt-get update && sudo apt-get install -y protobuf-compiler
    ```

## 1. Project Initialization

Initialize the project if you haven't already. This creates a `pyproject.toml` for dependency tracking.

```bash
uv init
```

## 2. Install Python Dependencies

We need to install TensorFlow with GPU support, legacy Keras support (for TF 2.16+), and other utilities.

*   **TensorFlow:** Use `tensorflow[and-cuda]` to bundle compatible NVIDIA libraries, avoiding system-level CUDA version mismatches.
*   **Keras:** Lc0 code relies on Keras 2 behaviors. TF 2.16+ defaults to Keras 3. We must install `tf_keras` and use an environment variable to force legacy mode.
*   **Protobuf:** Older Lc0 protos may have compatibility issues with Protobuf 4.x+, but modern TF requires newer versions. We usually target a version compatible with the generated code or regenerate it.

Run the following command to add all required dependencies:

```bash
uv add "tensorflow[and-cuda]" tf_keras pyyaml "protobuf<5.0" tensorflow-addons numpy
```

## 3. Compile Protobuf Definitions

The training pipeline uses Google Protobuf for data serialization. You must compile the `.proto` definition into Python code.

1.  Create the output directory if it doesn't exist:
    ```bash
    mkdir -p proto
    touch proto/__init__.py
    ```

2.  Compile `net.proto`:
    ```bash
    protoc -I=tf --python_out=proto tf/net.proto
    ```

## 4. Configuration (`example.yaml`)

The configuration file controls network architecture and training hyperparameters. It **must** match the architecture of the weights file you are fine-tuning from (if applicable).

**Key Parameters to Check:**
*   `input_type`: Matches the input planes of the network (e.g., `classic` for 112 planes, `canonical_v2` for others).
*   `embedding_style`: `new` vs `old`.
*   **Dimensions:** `embedding_size`, `policy_embedding_size`, `value_embedding_size`, `moves_left_embedding_size`.
*   **Auxiliary Heads:** `policy_d_aux`, `value_q`, `value_st`.

**Tip:** If you encounter "Shape mismatch" errors, use a script to inspect the `.pb.gz` file headers and weight sizes directly to deduce the correct parameters.

## 5. Running Training

To run the training script, you must set specific environment variables to ensure Python finds the modules and TensorFlow uses the correct Keras backend.

*   `TF_USE_LEGACY_KERAS=1`: Forces TensorFlow to use the `tf_keras` (Keras 2) backend. Critical for Lc0 code compatibility.
*   `PYTHONPATH=.`: Ensures the script can import modules (like `proto.net_pb2`) from the project root.

**Command:**

```bash
TF_USE_LEGACY_KERAS=1 PYTHONPATH=. uv run tf/train.py --cfg tf/configs/example.yaml
```

## Troubleshooting

### Out of Memory (OOM)
If you see `ResourceExhaustedError` or "Memory resources exhausted":
1.  **Reduce Batch Size:** Lower `batch_size` in your YAML config (e.g., from 512 to 128 or 64).
2.  **Reduce Shuffle Buffer:** Lower `shuffle_size` (e.g., to 50,000).
3.  **Check Zombie Processes:** Run `nvidia-smi` and `fuser -v /dev/nvidia0`. Kill any stuck python processes.

### Shape Mismatches
If loading weights fails with `KeyError: ... has wrong length`:
1.  Check the error message math. `protobuf size / output dim = input dim`.
2.  Adjust parameters in `example.yaml` (like `value_embedding_size`, `policy_d_aux`) to match the calculated dimensions.
3.  Verify `input_type` and `embedding_style`.

### "Unrecognized loss: future"
Remove experimental or deprecated loss keys (like `future`) from the `loss_weights` section of your YAML config.

### "Too many values to unpack"
The data parser might be returning extra fields (like future boards) that the training loop doesn't expect. Update `tf/tfprocess.py` unpacking lines (in `train_step`, `calculate_test_summaries`, etc.) to handle extra arguments:
```python
# Before
x, y, ... = next(iterator)
# After
x, y, ..., *args = next(iterator)
```
