
import numpy as np
import os

# Define paths
base_dir = "/home/abbad241/Desktop/PhD/Journals_Articles_Papers/Next paper/Lightweight_IA_V_3/Phase1/data/50s_window/sampling_1s"
val_path = os.path.join(base_dir, "val.npy")
test_path = os.path.join(base_dir, "test.npy")

# Load and count
try:
    if os.path.exists(val_path):
        val_data = np.load(val_path, allow_pickle=True)
        print(f"Validation Samples: {len(val_data)}")
    else:
        print("Validation file not found.")

    if os.path.exists(test_path):
        test_data = np.load(test_path, allow_pickle=True)
        print(f"Test Samples: {len(test_data)}")
    else:
        print("Test file not found.")

except Exception as e:
    print(f"Error: {e}")
