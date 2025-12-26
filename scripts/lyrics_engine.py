import os
import sys
import numpy as np
import faiss

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(ROOT_DIR)

from utils.lyrics_utils import embed_text

# --- Globals ---
LYRICS_INDEX = None
LYRICS_NAMES = None


def init_lyrics_resources():
    """Loads FAISS index and names once."""
    global LYRICS_INDEX, LYRICS_NAMES

    index_path = os.path.join(ROOT_DIR, "data", "lyrics_index.faiss")
    names_path = os.path.join(ROOT_DIR, "data", "lyrics_track_names.txt")

    if os.path.exists(index_path) and os.path.exists(names_path):
        LYRICS_INDEX = faiss.read_index(index_path)
        with open(names_path, "r", encoding="utf-8") as f:
            LYRICS_NAMES = [line.strip() for line in f]
        print("✅ [Lyrics Engine] Database Loaded")


# --- Main Export Function ---
def scan_lyrics(text):
    if LYRICS_INDEX is None: init_lyrics_resources()
    if LYRICS_INDEX is None: return []

    # 1. Embed
    q = embed_text(text).astype('float32').reshape(1, -1)
    faiss.normalize_L2(q)

    # 2. Search
    D, I = LYRICS_INDEX.search(q, k=3)

    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < len(LYRICS_NAMES):
            score = float(dist) * 100
            results.append({
                "song": LYRICS_NAMES[idx],
                "score": round(score, 2)
            })

    return results