import os
import sys
import numpy as np
import faiss

# --- 1. SETUP PATHS ---
# Current file is in: .../Music_Plagarism_Detector/scripts/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to get Project Root: .../Music_Plagarism_Detector/
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

from utils.lyrics_utils import embed_text

# Define folders
INDEX_PATH = os.path.join(ROOT_DIR, "data", "lyrics_index.faiss")
NAMES_PATH = os.path.join(ROOT_DIR, "data", "lyrics_track_names.txt")

# We want the top 3 matches
TOP_K = 3

# --- GLOBALS ---
LYRICS_INDEX = None
LYRICS_NAMES = None


def init_lyrics_resources():
    """Loads the FAISS index and song names list into global variables."""
    global LYRICS_INDEX, LYRICS_NAMES

    if not os.path.exists(INDEX_PATH) or not os.path.exists(NAMES_PATH):
        print(f"❌ Error: Database files missing at {INDEX_PATH}")
        return

    LYRICS_INDEX = faiss.read_index(INDEX_PATH)
    with open(NAMES_PATH, "r", encoding="utf-8") as f:
        LYRICS_NAMES = [line.strip() for line in f]
    print("✅ [Lyrics Engine] Database Loaded")


def query_text(text, index, names, top_k=TOP_K):
    """
    Embeds the text, normalizes it, and finds the top K matches in FAISS.
    """
    # 1. Embed text to vector
    q = embed_text(text).astype('float32').reshape(1, -1)

    # 2. Normalize (Critical for Percentage Accuracy)
    # This ensures the score is between 0.0 and 1.0
    faiss.normalize_L2(q)

    # 3. Search Index
    distances, indices = index.search(q, k=top_k)

    # 4. format results
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(names):
            # Convert 0-1 score to percentage
            percent = float(dist) * 100

            # Create the dictionary structure app.py expects
            results.append({
                "song": names[idx],
                "score": round(percent, 2)
            })
    return results


# --- MAIN EXPORT FUNCTION ---
def scan_lyrics(text):
    """
    This is the function app.py calls.
    """
    # Ensure resources are loaded
    if LYRICS_INDEX is None:
        init_lyrics_resources()

    if LYRICS_INDEX is None:
        return []  # Return empty if DB failed to load

    # Run the query
    return query_text(text, LYRICS_INDEX, LYRICS_NAMES)