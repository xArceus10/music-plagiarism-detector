import os
import sys
import numpy as np
import faiss
from tqdm import tqdm

# Add project root to path so we can import utils.*
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.lyrics_utils import embed_file

# --- Path Configuration ---
# Note: Changed to use ROOT_DIR for consistency with your other scripts.
LYRICS_DIR = os.path.join(ROOT_DIR, "data", "lyrics")
INDEX_PATH = os.path.join(ROOT_DIR, "data", "lyrics_index.faiss")
NAMES_PATH = os.path.join(ROOT_DIR, "data", "lyrics_track_names.txt")


def build_index():
    if not os.path.exists(LYRICS_DIR):
        print("No lyrics directory found:", LYRICS_DIR)
        return

    files = [f for f in os.listdir(LYRICS_DIR) if f.lower().endswith(".txt")]
    if not files:
        print("No lyrics files found in", LYRICS_DIR)
        return

    vectors = []
    names = []

    for fname in tqdm(files, desc="Embedding lyrics"):
        path = os.path.join(LYRICS_DIR, fname)
        try:
            vec = embed_file(path)  # Assumes this returns a 1D numpy array
            if vec.ndim == 2:  # Handle cases where embed_file might return [[...]]
                vec = vec.flatten()
            vectors.append(vec)
            names.append(fname)
        except Exception as e:
            print(f"Skipping {fname}: {e}")

    if not vectors:
        print("No embeddings created. Aborting.")
        return

    X = np.vstack(vectors).astype('float32')
    dim = X.shape[1]

    # -----------------------------------------------------------------
    # !!! HERE IS THE CRITICAL FIX !!!
    # You MUST normalize the vectors for IndexFlatIP to equal cosine similarity.
    print("Normalizing vectors for IndexFlatIP...")
    X_normalized = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    # -----------------------------------------------------------------

    index = faiss.IndexFlatIP(dim)

    # Add the NORMALIZED vectors to the index
    index.add(X_normalized)

    faiss.write_index(index, INDEX_PATH)

    with open(NAMES_PATH, "w", encoding="utf-8") as f:
        for n in names:
            f.write(n + "\n")

    print(f"\nSuccessfully built and saved lyrics index to {INDEX_PATH}")
    print(f"Saved {len(names)} track names to {NAMES_PATH}")


if __name__ == "__main__":
    build_index()