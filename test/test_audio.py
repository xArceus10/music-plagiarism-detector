

import os
import sys
import numpy as np
import faiss
import librosa
from scipy.spatial.distance import cdist

# Add project root to path so we can import utils.*
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.openl3_utils import extract_openl3_embedding  # must return normalized 1D np.array

# --- CONFIG ---
UPLOADS_DIR = os.path.join(ROOT_DIR, "data", "uploads")
PREVIEWS_DIR = os.path.join(ROOT_DIR, "data", "spotify_previews")
INDEX_PATH = os.path.join(ROOT_DIR, "data", "music_index.faiss")
EMBEDDINGS_PATH = os.path.join(ROOT_DIR, "data", "music_embeddings.npy")  # optional but recommended
TRACK_LIST_PATH = os.path.join(ROOT_DIR, "data", "track_names.txt")

TOP_K = 5
WEIGHT_AUDIO = 0.7
WEIGHT_MELODY = 0.3
Z_THRESHOLD = 2.0  # z > 2.0 => unusually similar (about top 2.5% if normal)

# --- Utilities ---


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:

    a = a.flatten()
    b = b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_index_and_metadata():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")
    index = faiss.read_index(INDEX_PATH)
    names = []
    if os.path.exists(TRACK_LIST_PATH):
        with open(TRACK_LIST_PATH, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
    else:
        # fallback: create fake names from index size
        names = [f"track_{i}" for i in range(index.ntotal)]
    return index, names





def melody_similarity(path_a: str, path_b: str, sr: int = 22050, hop_length: int = 512) -> float:

    try:
        y1, _ = librosa.load(path_a, sr=sr, mono=True)
        y2, _ = librosa.load(path_b, sr=sr, mono=True)

        # compute chroma (CQT-based is robust)
        C1 = librosa.feature.chroma_cqt(y=y1, sr=sr, hop_length=hop_length)
        C2 = librosa.feature.chroma_cqt(y=y2, sr=sr, hop_length=hop_length)

        # normalize per-frame
        C1 = C1 / (np.linalg.norm(C1, axis=0, keepdims=True) + 1e-8)
        C2 = C2 / (np.linalg.norm(C2, axis=0, keepdims=True) + 1e-8)

        # cost matrix: use cosine distance between chroma frames
        # cdist expects shape (frames, features)
        cost = cdist(C1.T, C2.T, metric="cosine")  # smaller = more similar

        # Use librosa's DTW to find minimum-cost alignment
        # librosa.sequence.dtw returns (D, wp); D is cumulative cost matrix
        D, wp = librosa.sequence.dtw(C=cost)
        # total cost is at [ -1, -1 ] of D
        total_cost = D[-1, -1]

        # normalize cost by path length to make comparable across tracks
        path_length = len(wp)
        avg_cost = total_cost / max(path_length, 1)

        # convert distance -> similarity (soft)
        # similarity = exp(-alpha * avg_cost) ; choose alpha to map typical costs to [0,1]
        # a stable conversion: similarity = 1 / (1 + avg_cost)
        sim = 1.0 / (1.0 + avg_cost)

        # clamp
        sim = float(np.clip(sim, 0.0, 1.0))
        return sim
    except Exception as e:
        # If melody extraction fails, return 0 and print debug msg
        print(f"Melody comparison failed between '{os.path.basename(path_a)}' and '{os.path.basename(path_b)}': {e}")
        return 0.0





def compute_dataset_stats(index, query_vec: np.ndarray, names_len_limit=1000):

    ntotal = index.ntotal
    if ntotal == 0:
        return 0.0, 1.0, np.array([0.0])

    # If index small, query all
    k_all = ntotal if ntotal <= names_len_limit else names_len_limit

    # If index supports search for k = k_all, we can use it
    # But when k_all < ntotal, we will query random subset by reconstructing some vectors from embeddings npy if exists.
    try:
        if k_all == ntotal:
            # Query all
            sims, ids = index.search(query_vec.reshape(1, -1).astype('float32'), k_all)
            sims = sims.flatten()
        else:
            # Prefer to load embeddings .npy if available for sampling
            if os.path.exists(EMBEDDINGS_PATH):
                all_emb = np.load(EMBEDDINGS_PATH)
                # ensure same dtype
                all_emb = all_emb.astype('float32')
                n = all_emb.shape[0]
                # sample indices uniformly (no replacement)
                idxs = np.random.choice(n, size=k_all, replace=False)
                sample = all_emb[idxs]
                # inner product when embeddings normalized == cosine
                sims = (sample @ query_vec).flatten()
            else:
                # fallback: request top k_all from index (gives top similarities, not distribution)
                sims, ids = index.search(query_vec.reshape(1, -1).astype('float32'), k_all)
                sims = sims.flatten()
    except Exception as e:
        print("Error computing dataset stats:", e)
        # fallback: return trivial stats
        return 0.0, 1.0, np.array([0.0])

    mean = float(np.mean(sims))
    std = float(np.std(sims)) if np.std(sims) > 0 else 1.0
    return mean, std, sims


# --- Main check function ---


def check_song(path_to_upload: str):
    if not os.path.exists(path_to_upload):
        print("Error: file does not exist:", path_to_upload)
        return

    # Extract OpenL3 embedding for query (must be normalized by extractor already)
    print("Extracting OpenL3 embedding...")
    q_emb = extract_openl3_embedding(path_to_upload).astype('float32')
    # normalize to be safe
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

    # Load FAISS index + metadata
    index, names = load_index_and_metadata()

    # Ensure index vectors are normalized (we assume this was done at build time)
    # Search top K
    print("Querying FAISS index...")
    k = min(max(TOP_K, 5), index.ntotal) if index.ntotal > 0 else TOP_K
    Dtop, Itop = index.search(q_emb.reshape(1, -1).astype('float32'), k)
    Dtop = Dtop.flatten()
    Itop = Itop.flatten()

    # Compute dataset mean/std for z-score (sample if needed)
    mean_sim, std_sim, all_sims = compute_dataset_stats(index, q_emb, names_len_limit=1000)

    # Gather candidates with melody sim and fused score
    results = []
    for rank, (sim_val, idx) in enumerate(zip(Dtop, Itop), start=1):
        if idx < 0 or idx >= len(names):
            name = f"track_{idx}"
        else:
            name = names[idx]

        # candidate preview path: try multiple variants
        candidate_file = name
        candidate_candidates = [
            os.path.join(PREVIEWS_DIR, candidate_file),
            os.path.join(PREVIEWS_DIR, candidate_file + ".mp3"),
            os.path.join(PREVIEWS_DIR, candidate_file.replace(" ", "_") + ".mp3"),
            os.path.join(PREVIEWS_DIR, candidate_file + ".MP3"),
        ]
        candidate_path = None
        for p in candidate_candidates:
            if os.path.exists(p):
                candidate_path = p
                break

        if candidate_path is None:
            # melody cannot be computed
            melody_sim = 0.0
        else:
            melody_sim = melody_similarity(path_to_upload, candidate_path)

        audio_sim = float(sim_val)  # inner product = cosine if DB normalized
        fused = WEIGHT_AUDIO * audio_sim + WEIGHT_MELODY * melody_sim

        results.append({
            "rank": rank,
            "name": name,
            "index": int(idx),
            "audio_sim": float(audio_sim),
            "melody_sim": float(melody_sim),
            "fused": float(fused),
            "candidate_path": candidate_path or ""
        })

    # Compute z-scores for fused values relative to dataset
    # For dataset fused distribution we approximate fused by combining audio_sim distribution (all_sims) with melody=0 mean,
    # because computing melody against all DB items is expensive. This gives a conservative baseline.
    # Optionally, you can compute melody for top-N DB items only for a refined z.
    # We'll compute z using audio_sim distribution for speed, then show fused as final score.
    top_fused = results[0]["fused"] if results else 0.0
    # approximate z with audio mean/std
    z_score = (top_fused - mean_sim) / (std_sim + 1e-12)

    # If you want a tighter z using fused on top-k only:
    fused_values = np.array([r["fused"] for r in results])
    fused_mean = float(np.mean(fused_values)) if fused_values.size > 0 else 0.0
    fused_std = float(np.std(fused_values)) if fused_values.size > 0 else 1.0
    fused_z = (top_fused - fused_mean) / (fused_std + 1e-12)

    # Print a human-readable report
    print("\nAlignment check: index contains", index.ntotal, "items")
    print("\nTop matches (audio + melody + fused):\n")
    for r in results:
        print(f"{r['rank']}. {r['name']}")
        print(f"   audio_sim (OpenL3 cosine) : {r['audio_sim']:.4f}")
        print(f"   melody_sim (chroma+DTW)  : {r['melody_sim']:.4f}")
        print(f"   fused_score (weighted)   : {r['fused']:.4f}")
        print(f"   candidate_path           : {r['candidate_path'] or 'N/A'}")
        print()

    print("Summary:")
    print(f"  Top fused score: {top_fused:.4f}")
    print(f"  Audio distribution mean/std (approx): {mean_sim:.4f} / {std_sim:.4f}")
    print(f"  Z-score (vs audio distribution)    : {z_score:.3f}")
    print(f"  Z-score (vs top-k fused)           : {fused_z:.3f}")

    if z_score > Z_THRESHOLD or fused_z > 2.0:
        print("\nDecision: Possible plagiarism — requires manual review.")
    elif top_fused > 0.85:
        print("\nDecision: High similarity (plausible plagiarism), manual review recommended.")
    elif top_fused > 0.7:
        print("\nDecision: Moderate similarity — suspicious, manual review recommended.")
    else:
        print("\nDecision: No critical similarity found.")

    print("\nDone.")


# --- CLI entrypoint ---
if __name__ == "__main__":
    if len(sys.argv) == 2:
        audio_path = sys.argv[1]
    else:
        mp3_files = [f for f in os.listdir(UPLOADS_DIR) if f.lower().endswith(".mp3")]
        if not mp3_files:
            print("No MP3 files found in uploads folder:", UPLOADS_DIR)
            sys.exit(1)
        audio_path = os.path.join(UPLOADS_DIR, mp3_files[0])
        print("No file passed; using first upload:", audio_path)

    check_song(audio_path)
