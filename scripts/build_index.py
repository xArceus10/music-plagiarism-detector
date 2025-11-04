

import os
import numpy as np
import faiss
# scripts/build_index.py

import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import numpy as np
import faiss
from utils.openl3_utils import extract_openl3_embedding

import os


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


SONGS_DIR = os.path.join(ROOT_DIR, "data", "spotify_previews")
INDEX_PATH = os.path.join(ROOT_DIR, "data", "music_index.faiss")
TRACK_LIST_PATH = os.path.join(ROOT_DIR, "data", "track_names.txt")
EMBEDDINGS_PATH = os.path.join(ROOT_DIR, "data", "music_embeddings.npy")


songs = [f for f in os.listdir(SONGS_DIR) if f.lower().endswith(".mp3")]
print(f" Found {len(songs)} songs in {SONGS_DIR}")

embeddings = []
track_names = []

for song in songs:
    full_path = os.path.join(SONGS_DIR, song)
    try:
        emb = extract_openl3_embedding(full_path).astype("float32")
        embeddings.append(emb)
        track_names.append(song)
        print(f" Processed: {song}")
    except Exception as e:
        print(f" Failed to process {song}: {e}")


if not embeddings:
    print(" No embeddings were extracted. Exiting.")
    exit(1)

all_embeddings = np.array(embeddings).astype("float32")


dims = all_embeddings.shape[1]
index = faiss.IndexFlatL2(dims)
index.add(all_embeddings)


faiss.write_index(index, INDEX_PATH)
print(f" Saved FAISS index to {INDEX_PATH}")


np.save(EMBEDDINGS_PATH, all_embeddings)
print(f" Saved embeddings to {EMBEDDINGS_PATH}")


with open(TRACK_LIST_PATH, "w", encoding="utf-8") as f:
    for name in track_names:
        f.write(name + "\n")
print(f" Saved track names to {TRACK_LIST_PATH}")
