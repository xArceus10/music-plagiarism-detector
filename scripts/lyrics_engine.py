import os
import sys
import numpy as np
import faiss

# --- SETUP PATHS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

from utils.lyrics_utils import embed_text

INDEX_PATH = os.path.join(ROOT_DIR, "data", "lyrics_index.faiss")
NAMES_PATH = os.path.join(ROOT_DIR, "data", "lyrics_track_names.txt")

# --- CONFIG: TOP 3 (Changed from 5) ---
TOP_K = 3  # <--- CHANGED HERE

LYRICS_INDEX = None
LYRICS_NAMES = None

def init_lyrics_resources():
    global LYRICS_INDEX, LYRICS_NAMES
    if not os.path.exists(INDEX_PATH) or not os.path.exists(NAMES_PATH):
        print(f"❌ Error: Database files missing at {INDEX_PATH}")
        return
    LYRICS_INDEX = faiss.read_index(INDEX_PATH)
    with open(NAMES_PATH, "r", encoding="utf-8") as f:
        LYRICS_NAMES = [line.strip() for line in f]
    print("✅ [Lyrics Engine] Database Loaded")

def query_text(text, index, names, top_k=TOP_K):
    q = embed_text(text).astype('float32').reshape(1, -1)
    faiss.normalize_L2(q)
    distances, indices = index.search(q, k=top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(names):
            percent = float(dist) * 100
            results.append({
                "song": names[idx],
                "score": round(percent, 2)
            })
    return results

def scan_lyrics(text):
    if LYRICS_INDEX is None: init_lyrics_resources()
    if LYRICS_INDEX is None: return []
    # Uses the updated TOP_K default (3)
    return query_text(text, LYRICS_INDEX, LYRICS_NAMES)