print("Script is starting...")
import os
import sys
import json
import numpy as np
import faiss

# --- Path Setup ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from scripts.build_index_chunked import process_file_into_chunks, INDEX_PATH, METADATA_PATH

UPLOADS_DIR = os.path.join(ROOT_DIR, "data", "uploads")


def check_song_chunks(audio_path):
    print(f"Checking chunks for: {os.path.basename(audio_path)}")

    if not os.path.exists(INDEX_PATH):
        print("Index not found. Run build_index_chunked.py first.")
        return {}

    # 1. Load Data
    try:
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, 'r') as f:
            meta_db = json.load(f)
    except Exception as e:
        print(f"Error loading index/metadata: {e}")
        return {}

    # 2. Chunk Upload
    q_vecs, q_metas = process_file_into_chunks(audio_path)
    if not q_vecs:
        print("No vectors extracted from audio.")
        return {}

    # 3. Search
    print(f"Searching {len(q_vecs)} chunks...")
    Q = np.array(q_vecs).astype('float32')
    D, I = index.search(Q, k=1)  # Top 1 match per chunk

    # 4. Vote
    scores = {}
    total_chunks = len(q_vecs)

    for i, (dist, idx) in enumerate(zip(D.flatten(), I.flatten())):
        if dist < 0.75: continue  # Filter noise

        match_name = meta_db[idx]['name']
        if match_name not in scores:
            scores[match_name] = {'count': 0, 'total_sim': 0.0}

        scores[match_name]['count'] += 1
        scores[match_name]['total_sim'] += float(dist)

    # 5. Format Results
    results = []
    for name, data in scores.items():
        # Coverage Score: Percentage of the upload that matched this song
        coverage = data['count'] / total_chunks
        # Similarity Score: Average cosine sim of the matched chunks
        avg_sim = data['total_sim'] / data['count']

        # Weighted Final Score (Heavy weight on coverage)
        final_score = (coverage * 0.8) + (avg_sim * 0.2)

        results.append({
            "name": name,
            "fused": float(np.clip(final_score, 0.0, 1.0)),  # Normalized 0-1
            "audio_sim": avg_sim,
            "coverage": coverage
        })

    results.sort(key=lambda x: x['fused'], reverse=True)

    top_match = results[0]['name'] if results else "No matches"

    return {
        "top_matches": results[:5],
        "top_fused_score": results[0]['fused'] if results else 0.0,
        "decision": f"Matched {top_match}" if results else "No matches"
    }


# --- THIS IS THE PART YOU WERE MISSING ---
if __name__ == "__main__":
    if not os.path.exists(UPLOADS_DIR):
        print(f"Error: Uploads folder not found at {UPLOADS_DIR}")
        sys.exit(1)

    mp3_files = [f for f in os.listdir(UPLOADS_DIR) if f.lower().endswith(".mp3")]

    if not mp3_files:
        print("No MP3 files found in uploads folder.")
    else:
        test_file = os.path.join(UPLOADS_DIR, mp3_files[0])
        print(f"Found file to test: {test_file}\n")

        result = check_song_chunks(test_file)

        print("\n--- JSON OUTPUT ---")
        print(json.dumps(result, indent=2))