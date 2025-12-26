import os
import sys
import json
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from scripts.check_audio_chunked import hybrid_check
from scripts.check_lyrics_similarity import query_file as check_lyrics

UPLOADS_DIR = os.path.join(ROOT_DIR, "data", "uploads")
WEIGHT_AUDIO = 0.6
WEIGHT_LYRICS = 0.4


def get_base_name(filename):
    return os.path.splitext(filename)[0]


def hybrid_check(audio_path, lyrics_path=None):
    print("🎧 Running CHUNKED Hybrid Check...")

    # 1. Run Chunked Audio Check
    audio_data = hybrid_check(audio_path)
    audio_matches = audio_data.get("top_matches", [])
    top_audio = audio_data.get("top_fused_score", 0.0)

    # 2. Run Lyric Check (Standard)
    lyrics_results = []
    if lyrics_path and os.path.exists(lyrics_path):
        try:
            lyrics_results = check_lyrics(lyrics_path)
        except:
            pass

    # 3. Fuse Results (Honest 0.0 Method)
    # Map Name -> Data
    audio_map = {get_base_name(m['name']): m['fused'] for m in audio_matches}
    lyric_map = {get_base_name(name): score for name, score in lyrics_results}

    all_names = set(audio_map.keys()) | set(lyric_map.keys())
    combined = []

    for base in all_names:
        a_score = audio_map.get(base, 0.0)
        l_score = lyric_map.get(base, 0.0)

        # Name recovery (try to get full filename)
        display_name = base
        for m in audio_matches:
            if get_base_name(m['name']) == base: display_name = m['name']

        hybrid = (a_score * WEIGHT_AUDIO) + (l_score * WEIGHT_LYRICS)

        combined.append({
            "song": display_name,
            "audio_score_percent": round(a_score * 100, 2),
            "lyrics_score_percent": round(l_score * 100, 2),
            "hybrid_score_percent": round(hybrid * 100, 2)
        })

    combined.sort(key=lambda x: x['hybrid_score_percent'], reverse=True)

    top_match = combined[0] if combined else None
    decision = "No Plagiarism Detected"
    if top_match and top_match['hybrid_score_percent'] > 80:
        decision = f"🔴 Potential Plagiarism: {top_match['song']}"
    elif top_match and top_match['hybrid_score_percent'] > 50:
        decision = f"🟠 Suspicious Similarity: {top_match['song']}"

    return {
        "top5_results": combined[:5],
        "hybrid_decision": decision,
        "hybrid_score_percent": top_match['hybrid_score_percent'] if top_match else 0,
        "audio_score_percent": round(top_audio * 100, 2),
        "lyrics_score_percent": round(lyrics_results[0][1] * 100, 2) if lyrics_results else 0
    }


if __name__ == "__main__":
    # Test block...
    pass