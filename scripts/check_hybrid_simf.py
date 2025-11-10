import os
import sys
import json
import numpy as np

# --- Path Setup ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# --- Import correct functions ---
from scripts.check_audio_sim import check_song as check_audio
from scripts.check_lyrics_similarity import query_file as check_lyrics

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


# --- NEW HELPER FUNCTION ---
def get_base_name(filename):
    """Removes .mp3, .txt, etc. from a filename."""
    return os.path.splitext(filename)[0]


def hybrid_check(audio_path, lyrics_path=None):
    print("\n🎧 Running HYBRID similarity check (Inflated Lyric Score Mode)")
    print("--------------------------------------------------")

    # --- Step 1: AUDIO SIMILARITY ---
    print("🔹 Checking audio similarity...")
    top_audio_score = 0.0
    audio_decision = "Audio check failed"
    audio_matches = []

    try:
        audio_results = check_audio(audio_path)
        if audio_results:
            top_audio_score = normalize_score(audio_results.get("top_fused_score", 0))
            audio_decision = audio_results.get("decision", "No decision")
            audio_matches = audio_results.get("top_matches", [])
        print(f"🎵 Top Audio (fused) score: {top_audio_score:.4f}")
    except Exception as e:
        print(f"❌ Audio similarity failed: {e}")

    # --- Step 2: LYRICS SIMILARITY ---
    print("\n📝 Checking lyrics similarity...")
    top_lyrics_score = 0.0
    lyrics_results = []

    if lyrics_path and os.path.exists(lyrics_path):
        try:
            lyrics_results = check_lyrics(lyrics_path)
            if lyrics_results:
                top_lyrics_score = normalize_score(lyrics_results[0][1])
        except Exception as e:
            print(f"⚠️ Lyrics similarity error: {e}")
    else:
        print("⚠️ No lyrics file provided — skipping lyrics similarity check.")
    print(f"📖 Top Lyrics score: {top_lyrics_score:.4f}")

    # --- Step 3: HYBRID FUSION (Corrected Logic) ---
    print("\n🔸 Combining results...")

    # Create lookup maps using the BASE NAME as the key
    audio_map = {get_base_name(m["name"]): {
        "score": normalize_score(m["fused"]),
        "original_name": m["name"]
    } for m in audio_matches}

    lyrics_map = {get_base_name(name): {
        "score": normalize_score(score),
        "original_name": name
    } for name, score in lyrics_results}

    # Get all unique base names
    all_base_names = set(audio_map.keys()) | set(lyrics_map.keys())

    combined_results = []
    for base_name in all_base_names:
        aud_data = audio_map.get(base_name)
        lyr_data = lyrics_map.get(base_name)

        # Get the honest score, or 0.0 if not found
        aud_score = aud_data["score"] if aud_data else 0.0
        lyr_score = lyr_data["score"] if lyr_data else 0.0

        # --- THIS IS THE FLAWED LOGIC YOU ASKED FOR ---
        # It copies the top lyric score to all audio matches
        if aud_score > 0 and lyr_score == 0.0:
            lyr_score = top_lyrics_score
        # But it does NOT copy the audio score (so .txt files get 0 audio)

        hybrid = WEIGHT_AUDIO * aud_score + WEIGHT_LYRICS * lyr_score

        # Prefer the .mp3 name, but fall back to the .txt name
        final_name = aud_data["original_name"] if aud_data else (lyr_data["original_name"] if lyr_data else base_name)

        combined_results.append({
            "song": final_name,
            "audio_score_percent": round(aud_score * 100, 2),
            "lyrics_score_percent": round(lyr_score * 100, 2),
            "hybrid_score_percent": round(hybrid * 100, 2)
        })

    # (Rest of the script is identical from here)
    if not combined_results:
        hybrid = WEIGHT_AUDIO * top_audio_score + WEIGHT_LYRICS * top_lyrics_score
        combined_results.append({
            "song": "No matching results",
            "audio_score_percent": round(top_audio_score * 100, 2),
            "lyrics_score_percent": round(top_lyrics_score * 100, 2),
            "hybrid_score_percent": round(hybrid * 100, 2)
        })

    combined_results.sort(key=lambda x: x["hybrid_score_percent"], reverse=True)
    top_result = combined_results[0]
    hybrid_percent = top_result["hybrid_score_percent"]

    # --- Step 4: DECISION LOGIC ---
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

    # --- Step 5: JSON OUTPUT (for Flask) ---
    result = {
        "audio_score_percent": round(top_audio_score * 100, 2),
        "lyrics_score_percent": round(top_lyrics_score * 100, 2),
        "hybrid_score_percent": hybrid_percent,
        "top_hybrid_match": top_result,
        "hybrid_decision": decision,
        "audio_only_decision": audio_decision,
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

    # --- THIS IS THE CRITICAL LINE FOR THE FLASK APP ---
    # Print a single line of JSON
    print(json.dumps(result))