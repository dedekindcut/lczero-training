import os
import yaml
import tensorflow as tf
import numpy as np
from tf import tfprocess # Corrected import path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=argparse.FileType("r"), required=True)
    parser.add_argument("--start", type=int, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.cfg.read())
    tfp = tfprocess.TFProcess(cfg)
    tfp.init_net()

    root_dir = os.path.join(cfg["training"]["path"], cfg["name"])
    checkpoint_path = os.path.join(root_dir, cfg["name"] + "-" + str(args.start))
    
    print(f"Loading {checkpoint_path}...")
    tfp.checkpoint.restore(checkpoint_path).expect_partial()

    print("\n--- Inspecting LoRA Weights ---")
    found_lora = False
    for var in tfp.model.variables:
        if "lora_A" in var.name:
            found_lora = True
            val = var.numpy()
            mean = np.mean(np.abs(val))
            max_val = np.max(np.abs(val))
            print(f"{var.name}: Mean Abs={mean:.6f}, Max={max_val:.6f}")
            if mean == 0:
                print("  WARNING: WEIGHTS ARE ALL ZERO!")
    
    if not found_lora:
        print("No LoRA variables found in model!")

if __name__ == "__main__":
    main()

