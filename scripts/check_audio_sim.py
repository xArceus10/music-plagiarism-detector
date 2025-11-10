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

from utils.openl3_utils import extract_openl3_embedding

# --- CONFIG ---
UPLOADS_DIR = os.path.join(ROOT_DIR, "data", "uploads")
PREVIEWS_DIR = os.path.join(ROOT_DIR, "data", "spotify_previews")
INDEX_PATH = os.path.join(ROOT_DIR, "data", "music_index.faiss")
TRACK_LIST_PATH = os.path.join(ROOT_DIR, "data", "track_names.txt")
STATS_PATH = os.path.join(ROOT_DIR, "data", "audio_stats.npz")

TOP_K = 5
WEIGHT_AUDIO = 0.7
WEIGHT_MELODY = 0.3
Z_THRESHOLD = 2.0
MELODY_CHECK_LIMIT = 5  # Only run slow DTW for the top N matches


# --- Utility Functions ---

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
        # print(f"Melody comparison failed between '{os.path.basename(path_a)}' and '{os.path.basename(path_b)}': {e}")
        return 0.0


def load_precomputed_stats():
    """Loads the precomputed mean and std for Z-score calculation."""
    if not os.path.exists(STATS_PATH):
        raise FileNotFoundError(f"Stats file not found: {STATS_PATH}.\n"
                                f"Please pre-compute and save your database mean/std for fast Z-scoring.")
    data = np.load(STATS_PATH)
    return float(data['mean_sim']), float(data['std_sim'])


def check_song(path_to_upload: str):
    if not os.path.exists(path_to_upload):
        print("Error: file does not exist:", path_to_upload)
        return

    print("Extracting OpenL3 embedding...")
    q_emb = extract_openl3_embedding(path_to_upload).astype('float32')
    # Normalize query vector (this is correct)
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

    # Load FAISS index and metadata
    index, names = load_index_and_metadata()



    if index.ntotal == 0:
        print("Error: FAISS index is empty.")
        return []

    print("Querying FAISS index...")
    k = min(max(TOP_K, 5), index.ntotal)
    Dtop, Itop = index.search(q_emb.reshape(1, -1).astype('float32'), k)
    Dtop = Dtop.flatten()  # Dtop should now be correct cosine similarity [0.0 to 1.0]
    Itop = Itop.flatten()

    # Load the PRECOMPUTED stats (FAST)
    mean_sim, std_sim = load_precomputed_stats()

    results = []
    top_fused = 0.0
    top_fused_name = "N/A"

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

        audio_sim = float(sim_val)
        melody_sim = 0.0

        # --- OPTIMIZATION: Conditional Melody Check ---
        if rank <= MELODY_CHECK_LIMIT and candidate_path:
            melody_sim = melody_similarity(path_to_upload, candidate_path)

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

    # --- NEW: Re-sort results based on the final fused score ---
    results.sort(key=lambda x: x["fused"], reverse=True)

    # Re-calculate top_fused and top_fused_name from the sorted list
    top_fused = results[0]["fused"] if results else 0.0
    top_fused_name = results[0]["name"] if results else "N/A"
    # Update ranks for printing
    for i, r in enumerate(results, 1):
        r["rank"] = i

    # --- Corrected Z-Score Logic ---
    # We must compare the top AUDIO score against the AUDIO distribution
    top_audio_sim = results[0]["audio_sim"] if results else 0.0

    # Calculate Z-score using the top AUDIO score, not the fused score
    z_score = (top_audio_sim - mean_sim) / (std_sim + 1e-12)

    print("\n🎧 Alignment check: index contains", index.ntotal, "items\n")





    print("Top matches (audio + melody + fused):\n")
    for r in results:
        melody_check_status = ""
        if r['melody_sim'] > 0 and r['rank'] <= MELODY_CHECK_LIMIT:
            melody_check_status = "(DTW Checked)"
        elif r['melody_sim'] == 0 and r['rank'] <= MELODY_CHECK_LIMIT:
            melody_check_status = "(DTW Failed/No Path)"
        elif r['rank'] > MELODY_CHECK_LIMIT:
            melody_check_status = "(DTW Skipped)"

        print(f"{r['rank']}. {r['name']} {melody_check_status}")
        print(f"   audio_sim (OpenL3 cosine): {r['audio_sim']:.4f}")
        print(f"   melody_sim (chroma+DTW):  {r['melody_sim']:.4f}")
        print(f"   fused_score (weighted):   {r['fused']:.4f}")

    print("\nSummary:")
    print(f"  Top fused score: {top_fused:.4f} (from {top_fused_name})")
    print(f"  Audio distribution mean/std: {mean_sim:.4f} / {std_sim:.4f}")
    print(f"  Z-score (vs audio distribution): {z_score:.3f}")

    # --- Decision thresholds ---
    decision = ""
    if top_fused >= 0.9 or z_score > Z_THRESHOLD:
        decision = f"🔴 Possible plagiarism — strong similarity with {top_fused_name}, manual review required."
        print("\nDecision: " + decision)
    elif top_fused >= 0.8:
        decision = f"🟠 High similarity — plausible plagiarism with {top_fused_name}, manual review recommended."
        print("\nDecision: " + decision)
    elif top_fused >= 0.7:
        decision = f"🟡 Moderate similarity — suspicious with {top_fused_name}, manual review advised."
        print("\nDecision: " + decision)
    else:
        decision = "🟢 No critical similarity detected."
        print("\nDecision: " + decision)

    print("\nDone.")

    # --- RETURN THE RESULTS ---
    return {
        "top_fused_score": top_fused,
        "mean_similarity": mean_sim,
        "std_similarity": std_sim,
        "z_score": z_score,
        "decision": decision,
        "top_matches": results
    }


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