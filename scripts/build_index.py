# scripts/build_index.py

import os
import numpy as np
import faiss
from utils.openl3_utils import extract_openl3_embedding

# Constants
PREVIEW_DIR = "data/spotify_previews"
INDEX_PATH = "data/music_index.faiss"
TRACK_LIST_PATH = "data/track_names.txt"
EMBEDDING_DIM = 512  # OpenL3 default for music

def build_index():
    print("[1] Scanning local previews...")

    # Collect all .mp3 files
    audio_files = [
        os.path.join(PREVIEW_DIR, f)
        for f in os.listdir(PREVIEW_DIR)
        if f.lower().endswith(".mp3")
    ]

    if not audio_files:
        print("❌ No preview files found in:", PREVIEW_DIR)
        return

    print(f"✅ Found {len(audio_files)} files. Extracting embeddings...")

    vectors = []
    names = []

    for path in audio_files:
        name = os.path.basename(path)
        print(f"🎵 Processing: {name}")

        try:
            emb = extract_openl3_embedding(path)
            vectors.append(emb)
            names.append(name)
        except Exception as e:
            print(f"⚠️ Skipping {name} — error: {e}")

    if not vectors:
        print("❌ No embeddings extracted. Aborting.")
        return

    print("[2] Building FAISS index...")
    vector_array = np.vstack(vectors).astype('float32')
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(vector_array)

    print(f"[3] Saving FAISS index → {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)

    print(f"[4] Saving track names → {TRACK_LIST_PATH}")
    with open(TRACK_LIST_PATH, "w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")

    print("✅ Done. Index built successfully.")

if __name__ == "__main__":
    build_index()
