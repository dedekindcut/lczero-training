import argparse
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "same") / w


def main():
    parser = argparse.ArgumentParser(description="Plot ELO ratings from a pickle file")
    parser.add_argument(
        "pickle_file", type=str, help="Path to the pickle file containing ELO data"
    )
    parser.add_argument(
        "--window", type=int, default=2, help="Moving average window size (default: 2)"
    )
    parser.add_argument(
        "--output", type=str, help="Output file path (if not specified, displays plot)"
    )

    args = parser.parse_args()

    try:
        with open(args.pickle_file, "rb") as f:
            data = pickle.load(f)

            if "elo" not in data:
                print("Error: Pickle file does not contain 'elo' key")
                sys.exit(1)

            elo = data["elo"]
            x = list(elo.keys())
            y = list(elo.values())
            y = moving_average(y, args.window)

            plt.figure(figsize=(10, 6))
            plt.plot(x, y)
            plt.xlabel("Step")
            plt.ylabel("ELO Rating")
            plt.title("ELO Rating over Time")
            plt.grid(True, alpha=0.3)

            if args.output:
                plt.savefig(args.output, dpi=300, bbox_inches="tight")
                print(f"Plot saved to {args.output}")
            else:
                plt.show()

    except FileNotFoundError:
        print(f"Error: File '{args.pickle_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
