# scripts/build_index.py

import os
import numpy as np
import faiss
from utils.spotify_downloader import fetch_and_download_previews
from utils.openl3_utils import extract_openl3_embedding

# Parameters
QUERY = "top hits"  # you can change this to any artist/genre/keyword
NUM_TRACKS = 25
PREVIEW_DIR = "data/spotify_previews"
INDEX_PATH = "data/music_index.faiss"
TRACK_LIST_PATH = "data/track_names.txt"

def build_index():
    print("[1] Fetching Spotify previews...")
    audio_files = fetch_and_download_previews(query=QUERY, limit=NUM_TRACKS, output_dir=PREVIEW_DIR)

    print("[2] Extracting embeddings...")
    vectors = []
    names = []

    for name, path in audio_files:
        print(f"🔎 Processing: {name} - {path}")
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            print(f"⚠️ File is missing or empty: {path}")
            continue
        try:
            emb = extract_openl3_embedding(path)
            vectors.append(emb)
            names.append(name)
        except Exception as e:
            print(f"❌ Skipping {name}: {e}")

    if not vectors:
        print("No embeddings created. Aborting index build.")
        return

    print("[3] Building FAISS index...")
    dimension = 512
    index = faiss.IndexFlatL2(dimension)
    vector_array = np.vstack(vectors).astype('float32')
    index.add(vector_array)

    print(f"[4] Saving FAISS index to {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)

    print(f"[5] Saving track names to {TRACK_LIST_PATH}")
    with open(TRACK_LIST_PATH, "w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")

    print("✅ Index build complete!")

if __name__ == "__main__":
    build_index()
