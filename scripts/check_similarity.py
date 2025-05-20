# scripts/check_similarity.py

import sys
import faiss
import numpy as np
import os
from utils.openl3_utils import extract_openl3_embedding

INDEX_PATH = "data/music_index.faiss"
TRACK_LIST_PATH = "data/track_names.txt"
TOP_K = 5  # Top 5 similar tracks to show
SIMILARITY_THRESHOLD = 0.85  # Cosine similarity threshold (0 to 1)

def cosine_similarity(v1, v2):
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
    return np.dot(v2_norm, v1_norm)

def check_song(path_to_upload):
    if not os.path.exists(INDEX_PATH):
        print("❌ Index not found. Please run build_index.py first.")
        return

    print(f"🎧 Analyzing: {path_to_upload}")
    try:
        query_emb = extract_openl3_embedding(path_to_upload).astype('float32')
    except Exception as e:
        print("❌ Failed to extract embedding:", e)
        return

    index = faiss.read_index(INDEX_PATH)
    with open(TRACK_LIST_PATH, "r", encoding="utf-8") as f:
        track_names = [line.strip() for line in f.readlines()]

    # L2 distance in FAISS → convert to cosine similarity manually
    distances, indices = index.search(np.array([query_emb]), k=TOP_K)
    candidates = indices[0]

    print("\n🎼 Similar Tracks Found:\n")
    for i, idx in enumerate(candidates):
        name = track_names[idx]
        dist = distances[0][i]
        sim = cosine_similarity(query_emb, index.reconstruct(idx))
        sim_score = float(sim)  # to make sure it's JSON/print safe

        if sim_score > SIMILARITY_THRESHOLD:
            print(f"🔴 {name} — Similarity: {sim_score:.2f}  ❗ Possible plagiarism")
        else:
            print(f"🟢 {name} — Similarity: {sim_score:.2f}")

    print("\n✅ Done.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_similarity.py <path_to_mp3>")
    else:
        check_song(sys.argv[1])
