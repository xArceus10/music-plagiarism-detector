import os
import sys
import json
import numpy as np
import faiss
import librosa
from scipy.spatial.distance import cdist

# --- PATH SETUP ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from scripts.build_index_chunked import process_file_into_chunks, INDEX_PATH, METADATA_PATH

# Configuration
UPLOADS_DIR = os.path.join(ROOT_DIR, "data", "uploads")
PREVIEWS_DIR = os.path.join(ROOT_DIR, "data", "songs")

# Weights for the Final Score
W_COVERAGE = 0.4  # How much of the song matched?
W_AUDIO_SIM = 0.3  # How close was the sound texture?
W_MELODY = 0.3  # Do the notes/melody align?

import difflib


def get_candidate_path(song_name):
    """
    Robustly attempts to find the actual MP3 file.
    1. Tries exact path.
    2. Tries appending extensions.
    3. Tries fuzzy matching against the directory listing.
    """
    # 1. Direct checks
    potential_paths = [
        os.path.join(PREVIEWS_DIR, song_name),
        os.path.join(PREVIEWS_DIR, f"{song_name}.mp3"),
        os.path.join(PREVIEWS_DIR, song_name.replace(" ", "_")),
        os.path.join(PREVIEWS_DIR, song_name.replace(" ", "_") + ".mp3"),
    ]

    for p in potential_paths:
        if os.path.exists(p):
            return p

    # 2. Fuzzy Match / Directory Scan
    # If the metadata name is "Adele Someone Like You" but file is "Adele_-_Someone_Like_You.mp3"
    try:
        files_in_dir = os.listdir(PREVIEWS_DIR)

        # specific fix for the ".." issue seen in your logs
        clean_search = song_name.replace("..", "").strip()

        # Find closest filename match in the folder
        matches = difflib.get_close_matches(clean_search, files_in_dir, n=1, cutoff=0.5)

        if matches:
            found_path = os.path.join(PREVIEWS_DIR, matches[0])
            print(f"   [Debug] Fuzzy matched '{song_name}' -> '{matches[0]}'")
            return found_path

    except Exception as e:
        print(f"   [Debug] Error scanning directory: {e}")

    return None
def calculate_dtw_melody(path_a, path_b, sr=22050):
    """
    Computes Dynamic Time Warping (DTW) similarity on Chroma features.
    Returns a score 0.0 to 1.0.
    """
    try:
        # Load audio (lightweight mono)
        y1, _ = librosa.load(path_a, sr=sr, mono=True, duration=30)
        y2, _ = librosa.load(path_b, sr=sr, mono=True, duration=30)

        # Extract Chroma (Pitch content)
        # We use CQT because it's pitch-invariant
        C1 = librosa.feature.chroma_cqt(y=y1, sr=sr)
        C2 = librosa.feature.chroma_cqt(y=y2, sr=sr)

        # Normalize
        C1 = librosa.util.normalize(C1)
        C2 = librosa.util.normalize(C2)

        # Compute Cost Matrix (Cosine distance)
        cost = cdist(C1.T, C2.T, metric='cosine')

        # Run DTW
        D, wp = librosa.sequence.dtw(C=cost)

        # Calculate similarity score
        # Lower cost = Higher similarity
        # Normalized cost is roughly between 0 (perfect) and 1 (bad)
        final_cost = D[-1, -1] / wp.shape[0]
        similarity = 1 - final_cost

        return float(np.clip(similarity, 0.0, 1.0))

    except Exception as e:
        print(f"Warning: Melody check failed for {os.path.basename(path_b)}: {e}")
        return 0.0


def hybrid_check(audio_path):
    print(f"\n--- Analyzing: {os.path.basename(audio_path)} ---")

    # 1. Load Resources
    if not os.path.exists(INDEX_PATH):
        print("Error: Index not found.")
        return

    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, 'r') as f:
        meta_db = json.load(f)

    # 2. Phase 1: Vector Search (Chunking)
    print("Step 1: Chunking & Vector Search...")
    q_vecs, _ = process_file_into_chunks(audio_path)

    if not q_vecs:
        print("Error: Could not extract chunks.")
        return

    Q = np.array(q_vecs).astype('float32')
    D, I = index.search(Q, k=1)  # Top match for each chunk

    # 3. Aggregate Votes
    candidates = {}
    total_chunks = len(q_vecs)

    for i, (dist, idx) in enumerate(zip(D.flatten(), I.flatten())):
        if dist < 0.65: continue  # Ignore weak chunk matches

        song_name = meta_db[idx]['name']

        if song_name not in candidates:
            candidates[song_name] = {
                'chunk_hits': 0,
                'accum_sim': 0.0
            }
        candidates[song_name]['chunk_hits'] += 1
        candidates[song_name]['accum_sim'] += float(dist)

    # Convert to list and sort by raw vote count
    top_candidates = []
    for name, data in candidates.items():
        coverage = data['chunk_hits'] / total_chunks
        avg_sim = data['accum_sim'] / data['chunk_hits']
        top_candidates.append({
            'name': name,
            'coverage': coverage,
            'audio_sim': avg_sim,
            'raw_score': (coverage * 0.7) + (avg_sim * 0.3)
        })

    # Keep only top 3 for the expensive check
    top_candidates.sort(key=lambda x: x['raw_score'], reverse=True)
    top_candidates = top_candidates[:3]

    print(f"Step 2: Melody Verification on top {len(top_candidates)} candidates...")

    # 4. Phase 2: Melody Verification (DTW)
    final_results = []

    for cand in top_candidates:
        candidate_path = get_candidate_path(cand['name'])

        melody_score = 0.0
        if candidate_path:
            melody_score = calculate_dtw_melody(audio_path, candidate_path)
        else:
            print(f"   [!] File not found for '{cand['name']}', skipping DTW.")

        # 5. Final Fuse
        # Formula: (Coverage * 0.4) + (AudioSim * 0.3) + (MelodyScore * 0.3)
        fused_score = (cand['coverage'] * W_COVERAGE) + \
                      (cand['audio_sim'] * W_AUDIO_SIM) + \
                      (melody_score * W_MELODY)

        final_results.append({
            "name": cand['name'],
            "fused_score": round(fused_score, 4),
            "details": {
                "coverage": round(cand['coverage'], 2),
                "audio_sim": round(cand['audio_sim'], 2),
                "melody_sim": round(melody_score, 2)
            }
        })
        print(f"   -> Checked '{cand['name']}': Melody Sim = {melody_score:.2f}")

    # Sort by final fused score
    final_results.sort(key=lambda x: x['fused_score'], reverse=True)

    # 6. Output Decision
    if not final_results:
        print("\nResult: No matches found.")
        return

    top = final_results[0]
    print("\n" + "=" * 40)
    print(f"TOP MATCH: {top['name']}")
    print(f"CONFIDENCE: {top['fused_score'] * 100:.1f}%")
    print("-" * 20)
    print("Breakdown:")
    print(f"   Match Coverage: {top['details']['coverage'] * 100:.0f}% (Chunks matched)")
    print(f"   Audio Texture:  {top['details']['audio_sim']:.2f}  (Vector similarity)")
    print(f"   Melody Structure: {top['details']['melody_sim']:.2f} (DTW Alignment)")
    print("=" * 40)

    # Output full JSON for API
    # print(json.dumps(final_results, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        # Auto-pick first file in uploads
        files = [f for f in os.listdir(UPLOADS_DIR) if f.endswith(".mp3")]
        if not files:
            print("No files in uploads.")
            sys.exit()
        target_file = os.path.join(UPLOADS_DIR, files[0])

    hybrid_check(target_file)