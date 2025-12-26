import os
import json
import random
import numpy as np
import faiss

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(ROOT_DIR, "data", "audio_chunked.faiss")
METADATA_PATH = os.path.join(ROOT_DIR, "data", "audio_chunked_meta.json")
OUTPUT_DATA_PATH = os.path.join(ROOT_DIR, "data", "audio_triplets.npz")


def create_triplets(num_triplets=2000):
    # 1. Validation
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        print("❌ Error: Index missing. Run scripts/2_build_audio_index.py first!")
        return

    print("🔹 Loading index to generate training data...")
    try:
        index = faiss.read_index(INDEX_PATH)
        # reconstruct_n pulls the raw vectors back out of FAISS
        all_vectors = index.reconstruct_n(0, index.ntotal)
    except Exception as e:
        print(f"❌ Error reading index: {e}")
        print("Tip: Make sure you are using faiss-cpu or faiss-gpu and the index is IndexFlatIP.")
        return

    # 2. Dimension Check (Critical!)
    # We expect 512 dimensions (Raw OpenL3).
    # If it is 128, it means you are trying to train on data that is ALREADY compressed.
    if all_vectors.shape[1] != 512:
        print(f"⚠️  WARNING: Your vectors are dimension {all_vectors.shape[1]}, but the model expects 512.")
        print(
            "   If you already trained the model, delete 'models/audio_adapter.pth' and 'data/audio_chunked.faiss' and start over.")
        # We allow it to proceed, but it might crash the training script later.

    with open(METADATA_PATH, 'r') as f:
        meta = json.load(f)

    # 3. Grouping
    print(f"🔹 Grouping {len(meta)} chunks by song...")
    song_to_indices = {}
    for idx, entry in enumerate(meta):
        name = entry['name']
        if name not in song_to_indices:
            song_to_indices[name] = []
        song_to_indices[name].append(idx)

    song_names = list(song_to_indices.keys())
    anchors, positives, negatives = [], [], []

    print(f"🔹 Generating {num_triplets} triplets (Self-Supervised)...")

    # 4. Sampling Loop
    attempts = 0
    while len(anchors) < num_triplets and attempts < num_triplets * 5:
        attempts += 1

        # Pick Anchor Song
        song_name = random.choice(song_names)
        indices = song_to_indices[song_name]

        # Need at least 2 chunks to form a Positive pair
        if len(indices) < 2: continue

        # A = Random Chunk
        # P = The very next chunk (Temporal consistency = Similarity)
        anchor_idx = random.choice(indices[:-1])
        positive_idx = anchor_idx + 1

        # N = Random Chunk from Random OTHER Song
        neg_song_name = random.choice(song_names)
        while neg_song_name == song_name:
            neg_song_name = random.choice(song_names)

        negative_idx = random.choice(song_to_indices[neg_song_name])

        anchors.append(all_vectors[anchor_idx])
        positives.append(all_vectors[positive_idx])
        negatives.append(all_vectors[negative_idx])

    # 5. Save
    np.savez(OUTPUT_DATA_PATH,
             anchors=np.array(anchors),
             positives=np.array(positives),
             negatives=np.array(negatives))

    print(f"✅ Success! Saved {len(anchors)} training triplets to {OUTPUT_DATA_PATH}")


if __name__ == "__main__":
    create_triplets()