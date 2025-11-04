
import os
import numpy as np
import faiss
from tqdm import tqdm
from utils.lyrics_utils import embed_file


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data"))
LYRICS_DIR = os.path.join(BASE_DIR, "lyrics")
INDEX_PATH = os.path.join(BASE_DIR, "lyrics_index.faiss")
NAMES_PATH = os.path.join(BASE_DIR, "lyrics_track_names.txt")

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
            vec = embed_file(path)
            vectors.append(vec)
            names.append(fname)
        except Exception as e:
            print(f"Skipping {fname}: {e}")

    if not vectors:
        print("No embeddings created. Aborting.")
        return

    X = np.vstack(vectors).astype('float32')
    dim = X.shape[1]


    index = faiss.IndexFlatIP(dim)
    index.add(X)

    faiss.write_index(index, INDEX_PATH)

    with open(NAMES_PATH, "w", encoding="utf-8") as f:
        for n in names:
            f.write(n + "\n")

    print("Saved lyrics index to", INDEX_PATH)
    print("Saved names to", NAMES_PATH)

if __name__ == "__main__":
    build_index()
