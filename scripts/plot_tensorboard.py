import os
import glob
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def plot_tensorboard_logs(logdir, output_dir="plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all event files
    event_files = glob.glob(os.path.join(logdir, "events.out.tfevents.*"))
    if not event_files:
        # Try recursive search
        event_files = glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True)
    
    if not event_files:
        print(f"No event files found in {logdir}")
        return

    print(f"Found {len(event_files)} event files.")
    
    # Dictionary to store data: tag -> (steps, values)
    data = defaultdict(lambda: {"steps": [], "values": []})

    for file_path in event_files:
        print(f"Processing {file_path}...")
        try:
            for e in tf.compat.v1.train.summary_iterator(file_path):
                for v in e.summary.value:
                    if v.tag:
                        # Extract scalar value
                        # v.simple_value is preferred for scalars, but sometimes it's in tensor
                        val = None
                        if v.HasField('simple_value'):
                            val = v.simple_value
                        elif v.HasField('tensor'):
                            try:
                                tensor = tf.make_ndarray(v.tensor)
                                if tensor.size == 1:
                                    val = float(tensor)
                            except Exception:
                                pass
                        
                        if val is not None:
                            data[v.tag]["steps"].append(e.step)
                            data[v.tag]["values"].append(val)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print(f"Extracted {len(data)} metrics.")

    # Plot each metric
    for tag, metric_data in data.items():
        steps = np.array(metric_data["steps"])
        values = np.array(metric_data["values"])

        # Sort by step
        sort_idx = np.argsort(steps)
        steps = steps[sort_idx]
        values = values[sort_idx]

        # Sanitize filename
        filename = tag.replace("/", "_").replace(" ", "_") + ".png"
        filepath = os.path.join(output_dir, filename)

        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, label=tag)
        plt.xlabel("Steps")
        plt.ylabel("Value")
        plt.title(tag)
        plt.legend()
        plt.grid(True)
        plt.savefig(filepath)
        plt.close()
        print(f"Saved plot for {tag} to {filepath}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_tensorboard.py <logdir>")
        sys.exit(1)
    
    log_dir = sys.argv[1]
    plot_tensorboard_logs(log_dir)
