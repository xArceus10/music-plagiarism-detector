import os
import sys
import numpy as np
import faiss

# Ensure utilities are in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.openl3_utils import extract_openl3_embedding

# --- CONFIGURATION ---
SONGS_DIR = os.path.join(ROOT_DIR, "data", "spotify_previews")
INDEX_PATH = os.path.join(ROOT_DIR, "data", "music_index.faiss")
TRACK_LIST_PATH = os.path.join(ROOT_DIR, "data", "track_names.txt")
EMBEDDINGS_PATH = os.path.join(ROOT_DIR, "data", "music_embeddings.npy")
STATS_PATH = os.path.join(ROOT_DIR, "data", "audio_stats.npz") # New: For Z-score data

# --- EMBEDDING EXTRACTION (No changes needed here) ---
songs = [f for f in os.listdir(SONGS_DIR) if f.lower().endswith(".mp3")]
print(f" Found {len(songs)} songs in {SONGS_DIR}")

embeddings = []
track_names = []

for song in songs:
    full_path = os.path.join(SONGS_DIR, song)
    try:
        # NOTE: Assuming extract_openl3_embedding returns un-normalized vector array here
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


# --------------------------------------------------------------------------
## 🚀 CORE FIXES FOR FASTER QUERIES AND CORRECT SCORING

# 1. NORMALIZE the embeddings (crucial step for cosine similarity)
print("\n1. Normalizing embeddings for FAISS (L2 norm)...")
normalized_embeddings = all_embeddings / (
    np.linalg.norm(all_embeddings, axis=1, keepdims=True) + 1e-12
)

# 2. Build IndexFlatIP (Inner Product)
# IP of normalized vectors == Cosine Similarity
dims = normalized_embeddings.shape[1]
index = faiss.IndexFlatIP(dims)
index.add(normalized_embeddings) # Add the normalized data

print(f" Saved FAISS index (IndexFlatIP) to {INDEX_PATH}")
faiss.write_index(index, INDEX_PATH)


# 3. Pre-calculate and save database statistics for FAST Z-scoring
print("\n3. Pre-calculating database statistics...")
mean_vector = np.mean(normalized_embeddings, axis=0)
# Calculate similarity of every vector to the mean vector (a proxy for database distribution)
sims_to_mean = normalized_embeddings @ mean_vector.T
mean_sim = float(np.mean(sims_to_mean))
std_sim = float(np.std(sims_to_mean))

print(f" Precomputed stats: Mean={mean_sim:.4f}, Std={std_sim:.4f}")
np.savez(STATS_PATH, mean_sim=mean_sim, std_sim=std_sim)
print(f" Saved stats to {STATS_PATH}")


# Saved un-normalized data for potential future use (e.g., re-indexing)
np.save(EMBEDDINGS_PATH, all_embeddings) 
print(f" Saved raw embeddings to {EMBEDDINGS_PATH}")


with open(TRACK_LIST_PATH, "w", encoding="utf-8") as f:
    for name in track_names:
        f.write(name + "\n")
print(f" Saved track names to {TRACK_LIST_PATH}")

print("\n🎉 Index build complete. **RERUN check_audio_sim.py NOW**.")