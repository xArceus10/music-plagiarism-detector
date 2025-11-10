import os
import sys
import json
import numpy as np
import faiss

# --- Path Setup ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# --- Import LOW-LEVEL functions ---
# We can't use the simple check_... scripts anymore
from utils.openl3_utils import extract_openl3_embedding
from utils.lyrics_utils import embed_text  # Assumes embed_text takes a string
from scripts.check_audio_sim import melody_similarity, PREVIEWS_DIR
from scripts.check_lyrics_similarity import load_index as load_lyrics_index
from scripts.check_audio_sim import load_index_and_metadata as load_audio_index

UPLOADS_DIR = os.path.join(ROOT_DIR, "data", "uploads")

# --- Weights ---
WEIGHT_AUDIO = 0.6
WEIGHT_LYRICS = 0.4


def normalize_score(score):
    """Ensure scores are within [0,1]."""
    try:
        return float(np.clip(score, 0.0, 1.0))
    except:
        return 0.0


def get_base_name(filename):
    """Removes .mp3, .txt, etc. from a filename."""
    return os.path.splitext(filename)[0]


def hybrid_check(audio_path, lyrics_path=None):
    print("\n🎧 Running HYBRID similarity check (SLOW, ACCURATE MODE)")
    print("--------------------------------------------------")

    # --- Step 1: Generate Query Embeddings ---
    print("🔹 Generating query embeddings...")
    # 1a. Audio
    q_emb_audio = extract_openl3_embedding(audio_path).astype('float32')
    q_emb_audio = q_emb_audio / (np.linalg.norm(q_emb_audio) + 1e-12)
    # 1b. Lyrics
    q_emb_lyrics = np.array([])
    if lyrics_path and os.path.exists(lyrics_path):
        with open(lyrics_path, "r", encoding="utf-8") as f:
            text = f.read()
        q_emb_lyrics = embed_text(text).astype('float32').reshape(1, -1)
        # Assumes your lyrics index is normalized, so normalize query
        q_emb_lyrics = q_emb_lyrics / (np.linalg.norm(q_emb_lyrics) + 1e-12)

    # --- Step 2: Load FAISS Indexes ---
    print("🔹 Loading FAISS indexes...")
    audio_index, audio_names = load_audio_index()
    lyrics_index, lyrics_names = load_lyrics_index()

    # This is a critical assumption for this to work
    if len(audio_names) != len(lyrics_names):
        print("❌ CRITICAL ERROR: Audio and Lyrics indexes are misaligned. Stopping.")
        return {}

    # --- Step 3: Run Top-K Searches ---
    print("🔹 Running Top-K searches...")
    k = 5
    # 3a. Audio Search
    D_audio, I_audio = audio_index.search(q_emb_audio.reshape(1, -1), k)
    # 3b. Lyrics Search
    D_lyrics, I_lyrics = np.array([]), np.array([])
    if q_emb_lyrics.size > 0:
        D_lyrics, I_lyrics = lyrics_index.search(q_emb_lyrics, k)
        D_lyrics = D_lyrics.flatten()
        I_lyrics = I_lyrics.flatten()

    # --- Step 4: Combine and Re-Compute ---
    print("🔸 Combining results and computing missing scores...")

    # We must use a dictionary to track unique songs by their index ID
    # This assumes index 42 in audio_names == index 42 in lyrics_names
    song_scores = {}

    def get_song(index_id):
        # Helper to initialize a song entry
        if index_id not in song_scores:
            song_scores[index_id] = {
                "song": audio_names[index_id],  # Use audio name by default
                "audio_score": 0.0,
                "melody_sim": 0.0,
                "lyrics_score": 0.0,
                "fused_audio": 0.0,
                "hybrid_score": 0.0
            }
        return song_scores[index_id]

    # 4a. Process Top Audio Matches
    print("   - Computing audio scores...")
    top_audio_score = 0.0
    for rank, (sim, idx) in enumerate(zip(D_audio.flatten(), I_audio.flatten())):
        song = get_song(idx)

        # --- Run slow melody check ---
        candidate_path = os.path.join(PREVIEWS_DIR, song["song"])
        if not os.path.exists(candidate_path):
            candidate_path = os.path.join(PREVIEWS_DIR, get_base_name(song["song"]) + ".mp3")  # Try base name

        melody_sim = melody_similarity(audio_path, candidate_path) if os.path.exists(candidate_path) else 0.0

        song["audio_score"] = normalize_score(sim)
        song["melody_sim"] = normalize_score(melody_sim)
        song["fused_audio"] = normalize_score(
            (0.7 * song["audio_score"]) + (0.3 * song["melody_sim"])
        )
        if rank == 0:
            top_audio_score = song["fused_audio"]

    # 4b. Process Top Lyrics Matches
    print("   - Computing lyrics scores...")
    top_lyrics_score = 0.0
    if q_emb_lyrics.size > 0:
        for rank, (sim, idx) in enumerate(zip(D_lyrics, I_lyrics)):
            song = get_song(idx)
            song["lyrics_score"] = normalize_score(sim)
            if rank == 0:
                top_lyrics_score = song["lyrics_score"]

    # 4c. --- THIS IS THE SLOW PART ---
    # Fill in the missing scores for all songs we've found
    print("   - Computing missing cross-scores (this is slow)...")
    for idx, song in song_scores.items():
        # If we have an audio score but no lyric score, compute it
        if song["audio_score"] > 0 and song["lyrics_score"] == 0 and q_emb_lyrics.size > 0:
            lyric_vec = lyrics_index.reconstruct(int(idx)).reshape(1, -1)
            missing_lyric_sim = float(np.dot(q_emb_lyrics, lyric_vec.T))
            song["lyrics_score"] = normalize_score(missing_lyric_sim)

        # If we have a lyric score but no audio score, compute it
        if song["lyrics_score"] > 0 and song["audio_score"] == 0:
            audio_vec = audio_index.reconstruct(int(idx)).reshape(1, -1)
            missing_audio_sim = float(np.dot(q_emb_audio.reshape(1, -1), audio_vec.T))
            song["audio_score"] = normalize_score(missing_audio_sim)
            # We skip the "fused" audio score for these, as it's too slow to run melody
            song["fused_audio"] = song["audio_score"]

            # Calculate final hybrid score
        song["hybrid_score"] = (WEIGHT_AUDIO * song["fused_audio"]) + (WEIGHT_LYRICS * song["lyrics_score"])

    # --- Step 5: Format and Return Results ---
    combined_results = [
        {
            "song": v["song"],
            "audio_score_percent": round(v["fused_audio"] * 100, 2),
            "lyrics_score_percent": round(v["lyrics_score"] * 100, 2),
            "hybrid_score_percent": round(v["hybrid_score"] * 100, 2)
        } for v in song_scores.values()
    ]

    if not combined_results:  # Fallback
        combined_results.append({
            "song": "No matching results",
            "audio_score_percent": 0.0,
            "lyrics_score_percent": 0.0,
            "hybrid_score_percent": 0.0
        })

    combined_results.sort(key=lambda x: x["hybrid_score_percent"], reverse=True)
    top_result = combined_results[0]
    hybrid_percent = top_result["hybrid_score_percent"]

    # --- Step 6: Decision Logic ---
    if hybrid_percent >= 90:
        decision = f"🔴 Strong similarity with '{top_result['song']}' — Possible plagiarism"
    elif hybrid_percent >= 80:
        decision = f"🟠 High similarity with '{top_result['song']}' — Manual review recommended"
    elif hybrid_percent >= 70:
        decision = f"🟡 Moderate similarity with '{top_result['song']}' — Possibly inspired"
    else:
        decision = f"🟢 Low similarity | Closest: '{top_result['song']}'"

    print("--------------------------------------------------")
    print(f"✅ Top Hybrid Score: {hybrid_percent:.2f}%")
    print(f"🧠 Decision: {decision}")
    print("--------------------------------------------------")

    # --- Step 7: JSON OUTPUT (for Flask) ---
    result = {
        "audio_score_percent": round(top_audio_score * 100, 2),
        "lyrics_score_percent": round(top_lyrics_score * 100, 2),
        "hybrid_score_percent": hybrid_percent,
        "top_hybrid_match": top_result,
        "hybrid_decision": decision,
        "audio_only_decision": "N/A (Slow Mode)",  # Audio-only decision is part of this script now
        "top5_results": combined_results[:5],
    }
    return result


if __name__ == "__main__":
    mp3_files = [f for f in os.listdir(UPLOADS_DIR) if f.lower().endswith(".mp3")]
    txt_files = [f for f in os.listdir(UPLOADS_DIR) if f.lower().endswith(".txt")]
    if not mp3_files:
        print("❌ No audio files found in uploads folder!")
        sys.exit(1)
    audio_path = os.path.join(UPLOADS_DIR, mp3_files[0])
    lyrics_path = os.path.join(UPLOADS_DIR, txt_files[0]) if txt_files else None
    print(f"Running in standalone mode...")
    print(f"Audio: {audio_path}")
    print(f"Lyrics: {lyrics_path if lyrics_path else 'None'}")
    result = hybrid_check(audio_path, lyrics_path)
    print(json.dumps(result))