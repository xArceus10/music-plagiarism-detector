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
EMBEDDINGS_PATH = os.path.join(ROOT_DIR, "data", "music_embeddings.npy")
TRACK_LIST_PATH = os.path.join(ROOT_DIR, "data", "track_names.txt")

TOP_K = 5
WEIGHT_AUDIO = 0.7
WEIGHT_MELODY = 0.3
Z_THRESHOLD = 2.0  # standard deviation threshold for outliers

# --- Utility Functions ---


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
        names = [f"track_{i}" for i in range(index.ntotal)]
    return index, names


def melody_similarity(path_a: str, path_b: str, sr: int = 22050, hop_length: int = 512) -> float:
    """Compare two songs using chroma features + DTW for melody similarity."""
    try:
        y1, _ = librosa.load(path_a, sr=sr, mono=True)
        y2, _ = librosa.load(path_b, sr=sr, mono=True)

        C1 = librosa.feature.chroma_cqt(y=y1, sr=sr, hop_length=hop_length)
        C2 = librosa.feature.chroma_cqt(y=y2, sr=sr, hop_length=hop_length)

        C1 = C1 / (np.linalg.norm(C1, axis=0, keepdims=True) + 1e-8)
        C2 = C2 / (np.linalg.norm(C2, axis=0, keepdims=True) + 1e-8)

        cost = cdist(C1.T, C2.T, metric="cosine")
        D, wp = librosa.sequence.dtw(C=cost)
        total_cost = D[-1, -1]
        path_length = len(wp)
        avg_cost = total_cost / max(path_length, 1)

        sim = 1.0 / (1.0 + avg_cost)
        return float(np.clip(sim, 0.0, 1.0))
    except Exception as e:
        print(f"Melody comparison failed between '{os.path.basename(path_a)}' and '{os.path.basename(path_b)}': {e}")
        return 0.0


def compute_dataset_stats(index, query_vec: np.ndarray, names_len_limit=1000):
    ntotal = index.ntotal
    if ntotal == 0:
        return 0.0, 1.0, np.array([0.0])

    k_all = ntotal if ntotal <= names_len_limit else names_len_limit

    try:
        if k_all == ntotal:
            sims, ids = index.search(query_vec.reshape(1, -1).astype('float32'), k_all)
            sims = sims.flatten()
        else:
            if os.path.exists(EMBEDDINGS_PATH):
                all_emb = np.load(EMBEDDINGS_PATH).astype('float32')
                n = all_emb.shape[0]
                idxs = np.random.choice(n, size=k_all, replace=False)
                sample = all_emb[idxs]
                sims = (sample @ query_vec).flatten()
            else:
                sims, ids = index.search(query_vec.reshape(1, -1).astype('float32'), k_all)
                sims = sims.flatten()
    except Exception as e:
        print("Error computing dataset stats:", e)
        return 0.0, 1.0, np.array([0.0])

    mean = float(np.mean(sims))
    std = float(np.std(sims)) if np.std(sims) > 0 else 1.0
    return mean, std, sims





def check_song(path_to_upload: str):
    if not os.path.exists(path_to_upload):
        print("Error: file does not exist:", path_to_upload)
        return

    print("Extracting OpenL3 embedding...")
    q_emb = extract_openl3_embedding(path_to_upload).astype('float32')
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

    # Load FAISS index and metadata
    index, names = load_index_and_metadata()

    # --- Normalize index vectors for cosine similarity ---
    print("Normalizing FAISS index vectors for cosine comparison...")
    xb = index.reconstruct_n(0, index.ntotal)
    xb = xb / (np.linalg.norm(xb, axis=1, keepdims=True) + 1e-12)
    new_index = faiss.IndexFlatIP(xb.shape[1])
    new_index.add(xb)
    index = new_index

    print("Querying FAISS index...")
    k = min(max(TOP_K, 5), index.ntotal) if index.ntotal > 0 else TOP_K
    Dtop, Itop = index.search(q_emb.reshape(1, -1).astype('float32'), k)
    Dtop = Dtop.flatten()
    Itop = Itop.flatten()

    mean_sim, std_sim, all_sims = compute_dataset_stats(index, q_emb, names_len_limit=1000)

    results = []
    for rank, (sim_val, idx) in enumerate(zip(Dtop, Itop), start=1):
        if idx < 0 or idx >= len(names):
            name = f"track_{idx}"
        else:
            name = names[idx]

        candidate_path = None
        for variant in [
            os.path.join(PREVIEWS_DIR, name),
            os.path.join(PREVIEWS_DIR, name + ".mp3"),
            os.path.join(PREVIEWS_DIR, name.replace(" ", "_") + ".mp3"),
            os.path.join(PREVIEWS_DIR, name + ".MP3"),
        ]:
            if os.path.exists(variant):
                candidate_path = variant
                break

        melody_sim = melody_similarity(path_to_upload, candidate_path) if candidate_path else 0.0
        audio_sim = float(sim_val)
        fused = WEIGHT_AUDIO * audio_sim + WEIGHT_MELODY * melody_sim

        results.append({
            "rank": rank,
            "name": name,
            "index": int(idx),
            "audio_sim": audio_sim,
            "melody_sim": melody_sim,
            "fused": fused,
            "candidate_path": candidate_path or ""
        })

    top_fused = results[0]["fused"] if results else 0.0
    z_score = (top_fused - mean_sim) / (std_sim + 1e-12)

    fused_values = np.array([r["fused"] for r in results])
    fused_mean = float(np.mean(fused_values)) if fused_values.size > 0 else 0.0
    fused_std = float(np.std(fused_values)) if fused_values.size > 0 else 1.0
    fused_z = (top_fused - fused_mean) / (fused_std + 1e-12)

    print("\nAlignment check: index contains", index.ntotal, "items\n")
    print("Top matches (audio + melody + fused):\n")
    for r in results:
        print(f"{r['rank']}. {r['name']}")
        print(f"   audio_sim (OpenL3 cosine): {r['audio_sim']:.4f}")
        print(f"   melody_sim (chroma+DTW):  {r['melody_sim']:.4f}")
        print(f"   fused_score (weighted):   {r['fused']:.4f}")
        print(f"   candidate_path: {r['candidate_path'] or 'N/A'}\n")

    print("Summary:")
    print(f"  Top fused score: {top_fused:.4f}")
    print(f"  Audio distribution mean/std: {mean_sim:.4f} / {std_sim:.4f}")
    print(f"  Z-score (vs audio distribution): {z_score:.3f}")
    print(f"  Z-score (vs top-k fused): {fused_z:.3f}")

    # --- Decision thresholds for cosine-scale scores ---
    if top_fused >= 0.9 or z_score > Z_THRESHOLD:
        print("\nDecision:  Possible plagiarism — strong similarity, manual review required.")
    elif top_fused >= 0.8:
        print("\nDecision: ⚠ High similarity — plausible plagiarism, manual review recommended.")
    elif top_fused >= 0.7:
        print("\nDecision: Moderate similarity — suspicious, manual review advised.")
    else:
        print("\nDecision:  No critical similarity detected.")

    print("\nDone.")



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
