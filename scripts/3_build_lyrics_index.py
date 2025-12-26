import os
import sys
import numpy as np
import faiss
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.lyrics_utils import embed_file

LYRICS_DIR = os.path.join(ROOT_DIR, "data", "lyrics")
INDEX_PATH = os.path.join(ROOT_DIR, "data", "lyrics_index.faiss")
NAMES_PATH = os.path.join(ROOT_DIR, "data", "lyrics_track_names.txt")

def build_index():
    if not os.path.exists(LYRICS_DIR):
        print(f"Missing lyrics folder: {LYRICS_DIR}"); return

    files = [f for f in os.listdir(LYRICS_DIR) if f.endswith(".txt")]
    vectors = []
    names = []

    for fname in tqdm(files, desc="Embedding lyrics"):
        try:
            vec = embed_file(os.path.join(LYRICS_DIR, fname))
            if vec.ndim > 1: vec = vec.flatten()
            vectors.append(vec)
            names.append(fname)
        except Exception as e:
            print(f"Skip {fname}: {e}")

    if not vectors: return

    # Normalize for Cosine Sim
    X = np.vstack(vectors).astype('float32')
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    faiss.write_index(index, INDEX_PATH)
    with open(NAMES_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(names))

    print(f"✅ Lyrics Index Built ({len(names)} tracks).")

if __name__ == "__main__":
    build_index()