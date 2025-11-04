import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.openl3_utils import extract_openl3_embedding
import faiss
import numpy as np

INDEX_PATH = os.path.join(ROOT_DIR, "data", "music_index.faiss")
TRACK_LIST_PATH = os.path.join(ROOT_DIR, "data", "track_names.txt")
EMBEDDINGS_PATH = os.path.join(ROOT_DIR, "data", "music_embeddings.npy")
TOP_K = 5
SIMILARITY_THRESHOLD = 0.85


def cosine_similarity(v1, v2):
    """Compute cosine similarity (without double-normalizing)."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def check_song(path_to_upload):
    if not os.path.exists(INDEX_PATH):
        print(" Index not found. Please run build_index.py first.")
        return

    if not os.path.exists(EMBEDDINGS_PATH):
        print(" Embeddings file not found. Please ensure 'music_embeddings.npy' exists.")
        return

    print(f" Analyzing: {path_to_upload}")
    try:
        query_emb = extract_openl3_embedding(path_to_upload).astype('float32')
    except Exception as e:
        print(" Failed to extract embedding:", e)
        return

    index = faiss.read_index(INDEX_PATH)
    all_embeddings = np.load(EMBEDDINGS_PATH)

    with open(TRACK_LIST_PATH, "r", encoding="utf-8") as f:
        track_names = [line.strip() for line in f.readlines()]

    # Debug check for alignment
    print(f"\n Alignment check: {len(track_names)} names, "
          f"{all_embeddings.shape[0]} embeddings, "
          f"{index.ntotal} in FAISS index")

    distances, indices = index.search(np.array([query_emb]), k=TOP_K)
    candidates = indices[0]

    print("\n Similar Tracks Found:\n")
    for i, idx in enumerate(candidates):
        name = track_names[idx]
        original_emb = all_embeddings[idx]

        # FAISS L2 distance (smaller is better)
        faiss_dist = distances[0][i]

        # Convert distance to similarity (optional)
        faiss_sim = 1 / (1 + faiss_dist)

        # Cosine similarity recomputed
        cos_sim = cosine_similarity(query_emb, original_emb)

        print(f"{i+1}. {name}")
        print(f"    FAISS distance: {faiss_dist:.4f}")
        print(f"    FAISS similarity: {faiss_sim:.4f}")
        print(f"    Cosine similarity: {cos_sim:.4f}")

        if cos_sim > SIMILARITY_THRESHOLD:
            print("    Possible plagiarism\n")
        else:
            print("    Likely original\n")

    print(" Done.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_similarity.py <path_to_mp3>")
    else:
        check_song(sys.argv[1])
