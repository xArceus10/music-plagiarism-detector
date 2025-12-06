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
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        print("❌ Error: Index missing. Run scripts/build_index_chunked.py first!")
        return

    print("🔹 Loading index to generate training data...")
    index = faiss.read_index(INDEX_PATH)

    # Extract all vectors from the index to use as training data
    try:
        # reconstruct_n pulls the raw vectors back out of FAISS
        all_vectors = index.reconstruct_n(0, index.ntotal)
    except:
        print("❌ Error: Your index type doesn't support reconstruction. Use IndexFlatIP.")
        return

    with open(METADATA_PATH, 'r') as f:
        meta = json.load(f)

    print(f"🔹 Grouping {len(meta)} chunks by song...")
    # Group chunk indices by song name so we know which chunks belong together
    song_to_indices = {}
    for idx, entry in enumerate(meta):
        name = entry['name']
        if name not in song_to_indices:
            song_to_indices[name] = []
        song_to_indices[name].append(idx)

    song_names = list(song_to_indices.keys())

    anchors = []
    positives = []
    negatives = []

    print(f"🔹 Generating {num_triplets} triplets (Self-Supervised)...")

    attempts = 0
    while len(anchors) < num_triplets and attempts < num_triplets * 5:
        attempts += 1

        # 1. Pick a random song (The Anchor Song)
        song_name = random.choice(song_names)
        indices = song_to_indices[song_name]

        # We need at least 2 chunks to make a pair
        if len(indices) < 2:
            continue

        # 2. Pick Anchor and Positive
        # Positive is the NEXT chunk in the sequence (high probability of similarity)
        anchor_idx = random.choice(indices[:-1])
        positive_idx = anchor_idx + 1

        # 3. Pick Negative (Random chunk from a DIFFERENT song)
        neg_song_name = random.choice(song_names)
        while neg_song_name == song_name:
            neg_song_name = random.choice(song_names)

        negative_idx = random.choice(song_to_indices[neg_song_name])

        anchors.append(all_vectors[anchor_idx])
        positives.append(all_vectors[positive_idx])
        negatives.append(all_vectors[negative_idx])

    # Save as numpy archive
    np.savez(OUTPUT_DATA_PATH,
             anchors=np.array(anchors),
             positives=np.array(positives),
             negatives=np.array(negatives))

    print(f"✅ Success! Saved {len(anchors)} training triplets to {OUTPUT_DATA_PATH}")


if __name__ == "__main__":
    create_triplets()