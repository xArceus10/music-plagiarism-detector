import numpy as np
import os

# --- Path Configuration ---
# This line is critical. It moves up two directories:
# 1. From the file to the 'scripts' directory.
# 2. From 'scripts' to the main 'Music_Plagarism_Detector' directory (the project root).
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBEDDINGS_PATH = os.path.join(ROOT_DIR, "data", "music_embeddings.npy")
STATS_PATH = os.path.join(ROOT_DIR, "data", "audio_stats.npz")

# --- Core Logic ---
try:
    print(f"Attempting to load embeddings from: {EMBEDDINGS_PATH}")
    all_emb = np.load(EMBEDDINGS_PATH).astype('float32')
except FileNotFoundError:
    print("\n🚨 CRITICAL ERROR: music_embeddings.npy not found.")
    print("Please ensure the 'data' folder is in the project root and that you have run your 'build_index' script to create the embeddings file.")
    exit(1)


# ... (rest of your logic for normalization and statistics calculation) ...

# Normalize them
all_emb = all_emb / (np.linalg.norm(all_emb, axis=1, keepdims=True) + 1e-12)

# Calculate the mean vector of all embeddings
mean_vector = np.mean(all_emb, axis=0)

# Get the distribution of similarities *against the mean vector*
sims_to_mean = all_emb @ mean_vector.T
mean_sim = float(np.mean(sims_to_mean))
std_sim = float(np.std(sims_to_mean))

print(f"\n✅ Precomputed stats: Mean={mean_sim:.4f}, Std={std_sim:.4f}")
np.savez(STATS_PATH, mean_sim=mean_sim, std_sim=std_sim)
print(f"Stats successfully saved to {STATS_PATH}")