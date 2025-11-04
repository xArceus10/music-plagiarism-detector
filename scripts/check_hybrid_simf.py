import os
import sys
import json
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from scripts.check_audio_sim import check_song as check_audio
from scripts.check_lyrics_similarity import query_file as check_lyrics

UPLOADS_DIR = os.path.join(ROOT_DIR, "data", "uploads")

# Weights for hybrid fusion
WEIGHT_AUDIO = 0.6
WEIGHT_LYRICS = 0.4


def normalize_score(score):
    """Ensure all scores are between 0 and 1."""
    try:
        return float(np.clip(score, 0.0, 1.0))
    except:
        return 0.0


def hybrid_check(audio_path, lyrics_path=None):
    print("\n🎧 Running HYBRID similarity check")
    print("--------------------------------------------------")

    # Step 1: Audio similarity
    print("🔹 Checking audio similarity...")
    from io import StringIO
    import contextlib

    audio_log = StringIO()
    with contextlib.redirect_stdout(audio_log):
        check_audio(audio_path)
    audio_output = audio_log.getvalue()

    top_audio_score = 0.0
    for line in audio_output.splitlines():
        if "Top fused score:" in line:
            try:
                top_audio_score = float(line.split(":")[1].strip())
            except:
                pass

    top_audio_score = normalize_score(top_audio_score)
    print(f"🎵 Audio similarity score: {top_audio_score:.4f}")

    # Step 2: Lyrics similarity
    top_lyrics_score = 0.0
    if lyrics_path and os.path.exists(lyrics_path):
        print("\n📝 Checking lyrics similarity...")
        try:
            results = check_lyrics(lyrics_path)
            if results and isinstance(results, list):
                top_lyrics_score = float(results[0][1])
                top_lyrics_score = normalize_score(top_lyrics_score)
        except Exception as e:
            print(f"⚠️ Lyrics similarity error: {e}")
        print(f"📖 Lyrics similarity score: {top_lyrics_score:.4f}")
    else:
        print("\n⚠️ No lyrics file provided — skipping lyrics similarity check.")

    # Step 3: Weighted fusion
    print("\n🔸 Combining results...")
    hybrid_score = WEIGHT_AUDIO * top_audio_score + WEIGHT_LYRICS * top_lyrics_score

    # Convert to percentages
    audio_percent = round(top_audio_score * 100, 2)
    lyrics_percent = round(top_lyrics_score * 100, 2)
    hybrid_percent = round(hybrid_score * 100, 2)

    # Step 4: Decision logic
    if hybrid_percent >= 90:
        decision = "Strong similarity | Possible plagiarism"
    elif hybrid_percent >= 80:
        decision = "High similarity | Manual review recommended"
    elif hybrid_percent >= 70:
        decision = "Moderate similarity | Possibly inspired"
    else:
        decision = "Low similarity | Unlikely plagiarism"

    print("--------------------------------------------------")
    print(f"✅ Final HYBRID Score: {hybrid_percent:.2f}%")
    print(f"🧠 Decision: {decision}")
    print("--------------------------------------------------")

    # Step 5: Safe JSON result (frontend-ready)
    result = {
        "audio_score_percent": float(audio_percent),
        "lyrics_score_percent": float(lyrics_percent),
        "hybrid_score_percent": float(hybrid_percent),
        "decision": decision
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

    result = hybrid_check(audio_path, lyrics_path)
    print("\nFinal JSON Output:\n", json.dumps(result, indent=4))
