"""
Optional: Add this at the top of run_qd_grasp.py's main() function to make runs reproducible.

Usage:
    1. Import this module in run_qd_grasp.py
    2. Call set_seed(42) at the start of main()
    3. Every run will now use the same initial random positions
"""

import numpy as np
import random


def set_seed(seed_value):
    """Set random seed for reproducibility."""
    np.random.seed(seed_value)
    random.seed(seed_value)
    print(f"\n{'='*60}")
    print(f"RANDOM SEED SET TO: {seed_value}")
    print(f"All runs will now be reproducible with this seed.")
    print(f"{'='*60}\n")


# Example usage:
# from set_random_seed import set_seed
# set_seed(42)  # Use any integer you want


