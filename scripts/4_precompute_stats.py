import os
import sys
import numpy as np
import faiss

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(ROOT_DIR, "data", "audio_chunked.faiss")
STATS_PATH = os.path.join(ROOT_DIR, "data", "audio_stats.npz")


def compute():
    if not os.path.exists(INDEX_PATH):
        print("Index not found. Run Step 2 first.");
        return

    print("Loading Index...")
    index = faiss.read_index(INDEX_PATH)

    # 1. Reconstruct Vectors
    try:
        all_vecs = index.reconstruct_n(0, index.ntotal)
    except:
        print("Error: Index type does not support reconstruction.");
        return

    # 2. Calculate Global Mean Vector
    mean_vec = np.mean(all_vecs, axis=0)

    # 3. Compute distribution of similarities (to the mean)
    # This tells us: "How close is a random song to the average sound?"
    sims = np.dot(all_vecs, mean_vec)

    mu = np.mean(sims)
    sigma = np.std(sims)

    print(f"Stats: Mean Sim = {mu:.4f}, Std Dev = {sigma:.4f}")
    np.savez(STATS_PATH, mean_sim=mu, std_sim=sigma)
    print("✅ Stats saved.")


if __name__ == "__main__":
    compute()