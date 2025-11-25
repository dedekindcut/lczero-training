import os
import yaml
import tensorflow as tf
import numpy as np
import argparse
from tf.chunkparser import ChunkParser
import glob

def main():
    # Setup standard config mock
    chunks = glob.glob("/workspace/data/QueenOddsV2/**/*.gz", recursive=True)[:5] # Check first 5 files in QueenOddsV2
    if not chunks:
        # Fallback to non-recursive if structure is flat
        chunks = glob.glob("/workspace/data/QueenOddsV2/*.gz")[:5]
    
    if not chunks:
        print("No chunks found!")
        return

    print(f"Checking {len(chunks)} chunks...")

    # Initialize Parser
    # input_format=1 is INPUT_CLASSICAL_112_PLANE
    parser = ChunkParser(chunks, 1, shuffle_size=1, batch_size=10, workers=1) 

    # Create dataset
    ds = tf.data.Dataset.from_generator(parser.parse, output_types=(tf.string,) * 9)
    
    # Iterate a few batches
    count = 0
    w_wins = 0
    b_wins = 0
    draws = 0
    
    print("\n--- Inspecting Parsed Results ---")
    
    # We need to manually unpack the binary string like parse_function does in chunkparsefunc.py
    # But parse_function is complex. 
    # Let's just rely on what the parser yields? 
    # Wait, parser.parse yields raw bytes (chunks of records).
    
    # Actually, let's use the parse_function if possible, or just decode the V6 struct manually again here.
    # To keep it simple, let's look at what chunkparser yields.
    # It yields a tuple of 9 strings.
    # 0: input_planes
    # 1: policy_target
    # 2: value_target (THIS IS WHAT WE NEED)
    # ...
    
    from tf.chunkparsefunc import parse_function
    
    ds = ds.map(parse_function)
    
    for batch in ds.take(5): # Take 5 batches
        # Unpack the 9-tuple returned by chunkparsefunc.py
        (planes, probs, winner, q, plies_left, st_q, opp_probs, next_probs, fut) = batch
        
        winner_vals = winner.numpy()
        q_vals = q.numpy()
        
        print("\n--- Batch ---")
        print(f"Winner (WDL): {winner_vals}")
        print(f"Q (WDL?): {q_vals}")
        
        for v in winner_vals:
            # Assuming WDL format [Win, Draw, Loss]
            if np.argmax(v) == 0: w_wins += 1
            elif np.argmax(v) == 1: draws += 1
            else: b_wins += 1

    print(f"\nStats from sample: W={w_wins}, D={draws}, L={b_wins}")

if __name__ == "__main__":
    main()
