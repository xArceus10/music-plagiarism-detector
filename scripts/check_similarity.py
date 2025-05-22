import sys
import os
import numpy as np
import faiss

# Configure paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.openl3_utils import extract_openl3_embedding

# Constants
INDEX_PATH = os.path.join(ROOT_DIR, "data", "music_index.faiss")
TRACK_LIST_PATH = os.path.join(ROOT_DIR, "data", "track_names.txt")
TOP_K = 5
SIMILARITY_THRESHOLD = 0.85


def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors"""
    # Ensure 1D arrays and handle zero vectors
    v1 = v1.flatten()
    v2 = v2.flatten()
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.dot(v1, v2) / norm_product if norm_product != 0 else 0.0


def check_song(path_to_upload):
    if not os.path.exists(INDEX_PATH):
        print("❌ Index not found. Please run build_index.py first.")
        return

    print(f"🎧 Analyzing: {os.path.basename(path_to_upload)}")

    try:
        # Extract and validate embedding
        query_emb = extract_openl3_embedding(path_to_upload)
        query_emb = query_emb.astype(np.float32).squeeze()  # Ensure 1D array
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return

    try:
        index = faiss.read_index(INDEX_PATH)
        with open(TRACK_LIST_PATH, "r", encoding="utf-8") as f:
            track_names = [line.strip() for line in f]
    except Exception as e:
        print(f"❌ Index loading failed: {e}")
        return

    # FAISS search
    distances, indices = index.search(query_emb.reshape(1, -1), TOP_K)

    print("\n🎼 Similar Tracks Found:\n")

    for idx in indices[0]:
        try:
            # Convert numpy index to Python int
            idx_int = int(idx)
            name = track_names[idx_int]

            # Reconstruct vector PROPERLY
            reconstructed_vec = np.zeros(index.d, dtype=np.float32)
            index.reconstruct(idx_int, reconstructed_vec)

            # Calculate similarity
            sim = cosine_similarity(query_emb, reconstructed_vec)
            sim_score = max(0.0, min(float(sim), 1.0))  # Clamp between 0-1

            # Format output
            if sim_score > SIMILARITY_THRESHOLD:
                print(f"🔴 {name} — Similarity: {sim_score:.2f} ❗ Possible plagiarism")
            else:
                print(f"🟢 {name} — Similarity: {sim_score:.2f}")

        except Exception as e:
            print(f"⚠️ Error processing index {idx}: {e}")

    print("\n✅ Analysis complete")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_similarity.py <path_to_mp3>")
        sys.exit(1)

    check_song(sys.argv[1])