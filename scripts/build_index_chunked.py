import os
import sys
import numpy as np
import faiss
import json

# --- Path Setup ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.openl3_utils import extract_openl3_embedding

# --- Config ---
SONGS_DIR = os.path.join(ROOT_DIR, "data", "songs")
INDEX_PATH = os.path.join(ROOT_DIR, "data", "audio_chunked.faiss")
METADATA_PATH = os.path.join(ROOT_DIR, "data", "audio_chunked_meta.json")

CHUNK_SIZE = 10.0  # Seconds
HOP_SIZE = 5.0  # Seconds (Overlap)


def process_file_into_chunks(filepath):
    try:
        # extract_openl3_embedding returns shape (T, 512) where T is seconds
        full_emb = extract_openl3_embedding(filepath)

        # Handle short files
        if full_emb.ndim == 1:
            full_emb = full_emb.reshape(1, -1)

        vectors = []
        metadata = []
        filename = os.path.basename(filepath)
        num_seconds = full_emb.shape[0]

        # Sliding window
        for start_sec in range(0, num_seconds, int(HOP_SIZE)):
            end_sec = start_sec + int(CHUNK_SIZE)
            if end_sec > num_seconds and start_sec > 0: break  # Skip partial end chunks if we have others

            # Slice and Average
            chunk_vectors = full_emb[start_sec:end_sec]
            chunk_vec = np.mean(chunk_vectors, axis=0)

            # Normalize
            chunk_vec = chunk_vec / (np.linalg.norm(chunk_vec) + 1e-12)

            vectors.append(chunk_vec)
            metadata.append({
                "name": filename,  # Using 'name' to match your other scripts
                "time": f"{start_sec}-{end_sec}s"
            })

        return vectors, metadata
    except Exception as e:
        print(f"Error chunking {filepath}: {e}")
        return [], []


def build():
    all_vecs = []
    all_meta = []

    files = [f for f in os.listdir(SONGS_DIR) if f.lower().endswith(".mp3")]
    print(f"Found {len(files)} songs. Chunking...")

    for i, f in enumerate(files):
        print(f"[{i + 1}/{len(files)}] {f}...")
        v, m = process_file_into_chunks(os.path.join(SONGS_DIR, f))
        all_vecs.extend(v)
        all_meta.extend(m)

    if not all_vecs: return

    # Save FAISS
    X = np.array(all_vecs).astype('float32')
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, INDEX_PATH)

    # Save Metadata
    with open(METADATA_PATH, 'w') as f:
        json.dump(all_meta, f)

    print(f"✅ Indexed {len(all_meta)} chunks.")


if __name__ == "__main__":
    build()