import sys
import os
import gzip
import numpy as np

# Dynamically add current directory to sys.path to find 'proto' package
# Assuming 'proto' directory is in the same directory as this script or in project root.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if "dm-lczero-training" in project_root: # Adjust this check if your project root differs
    sys.path.insert(0, project_root)
else:
    sys.path.insert(0, script_dir)

try:
    import proto.net_pb2 as pb
except ImportError:
    print("Error: Could not import 'proto.net_pb2'.")
    print("Ensure 'proto' directory with 'net_pb2.py' and '__init__.py' is in your PYTHONPATH or current directory (or project root).")
    sys.exit(1)

# Copied/adapted from tf/net.py::Net.denorm_layer_v2
def denorm_layer_v2(layer_pb):
    if not hasattr(layer_pb, 'params') or not layer_pb.params:
        return np.array([]) # Empty weights
    params = np.frombuffer(layer_pb.params, np.uint16).astype(np.float32)
    params /= 0xffff
    return params * (layer_pb.max_val - layer_pb.min_val) + layer_pb.min_val

def inspect_pb_gz(filepath):
    print(f"--- Inspecting Protobuf GZ file: {filepath} ---")
    net_pb = pb.Net()
    try:
        with gzip.open(filepath, 'rb') as f:
            net_pb.ParseFromString(f.read())
    except Exception as e:
        print(f"Failed to parse {filepath}: {e}")
        return

    # Accessing enums directly from the module or the message class, depending on generation style.
    # Usually: pb.NetworkFormat.Name(value) or pb.NetworkFormat.Value.Name(value)
    # The structure in net_pb2.py for older protoc/versions often puts enums at module level or as class attributes.
    
    # Let's try to just print the integer values first to be safe, or use a safer lookup.
    print(f"Network Format: {net_pb.format.network_format.network}")
    print(f"Input Format: {net_pb.format.network_format.input}")
    print(f"Policy Format: {net_pb.format.network_format.policy}")
    print(f"Value Format: {net_pb.format.network_format.value}")
    print(f"Moves Left Format: {net_pb.format.network_format.moves_left}")
    print(f"Headcount: {net_pb.weights.headcount}")
    print(f"Pol Headcount: {net_pb.weights.pol_headcount}")

    print("\n--- Detailed Weights ---")

    # This recursive helper function traverses the protobuf message and prints details for fields that look like weights.
    def print_weights(obj, current_path=""):
        if obj is None:
            return

        # Use an internal list of fields that represent actual weights
        # These are typically sub-messages that have 'params', 'min_val', 'max_val'
        
        for field_descriptor in obj.DESCRIPTOR.fields:
            field_name = field_descriptor.name
            full_field_path = f"{current_path}{field_name}"
            value = getattr(obj, field_name)

            if field_descriptor.type == field_descriptor.TYPE_MESSAGE:
                if field_descriptor.label == field_descriptor.LABEL_REPEATED:
                    for i, item in enumerate(value):
                        print_weights(item, f"{full_field_path}[{i}]/")
                else:
                    # Check if this nested message itself is a 'weight container'
                    if hasattr(value, 'params') and hasattr(value, 'min_val') and hasattr(value, 'max_val'):
                        decoded_weights = denorm_layer_v2(value)
                        print(f"{full_field_path}: Shape={decoded_weights.shape}, Size={decoded_weights.size}")
                    else:
                        print_weights(value, f"{full_field_path}/")

    print_weights(net_pb.weights, current_path="net.weights/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_pb_gz.py <path_to_pb_gz_file>")
        sys.exit(1)
    
    pb_gz_file = sys.argv[1]
    inspect_pb_gz(pb_gz_file)
