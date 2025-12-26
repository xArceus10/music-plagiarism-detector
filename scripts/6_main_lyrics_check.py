import os
import sys
import numpy as np
import faiss

# --- 1. SETUP PATHS ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.lyrics_utils import embed_text

# Define folders
UPLOADS_DIR = os.path.join(ROOT_DIR, "data", "uploads")  # Input folder
INDEX_PATH = os.path.join(ROOT_DIR, "data", "lyrics_index.faiss")
NAMES_PATH = os.path.join(ROOT_DIR, "data", "lyrics_track_names.txt")

# We want the top 3 matches
TOP_K = 3


def load_index():
    """Loads the FAISS index and song names list."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(NAMES_PATH):
        print("❌ Error: Database files missing. Run '4_build_lyrics_database.py' first.")
        sys.exit(1)

    index = faiss.read_index(INDEX_PATH)
    with open(NAMES_PATH, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f]
    return index, names


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
            results.append((names[idx], float(dist)))
    return results


if __name__ == "__main__":
    # 1. Load Database
    print("⏳ Loading Lyrics Database...")
    index, db_names = load_index()
    print("✅ Database Loaded.\n")

    # 2. Check Uploads Folder
    if not os.path.exists(UPLOADS_DIR):
        print(f"❌ Uploads folder not found: {UPLOADS_DIR}")
        sys.exit(1)

    txt_files = [f for f in os.listdir(UPLOADS_DIR) if f.lower().endswith(".txt")]

    if not txt_files:
        print("❌ No .txt files found in 'data/uploads'")
        sys.exit(0)

    print(f"🔎 Analyzing {len(txt_files)} file(s)...\n")

    # 3. Process Each File
    for txt_file in txt_files:
        txt_path = os.path.join(UPLOADS_DIR, txt_file)

        # Read the lyrics file
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Get Matches
        matches = query_text(content, index, db_names)

        # Print Report
        print("=" * 50)
        print(f"📄 INPUT: {txt_file}")
        print("=" * 50)
        print(f"{'RANK':<5} | {'MATCHED SONG':<30} | {'SIMILARITY'}")
        print("-" * 55)

        for i, (song_name, score) in enumerate(matches, 1):
            # Convert 0-1 score to percentage
            percent = score * 100
            print(f"#{i:<4} | {song_name[:28]:<30} | {percent:.2f}%")

        print("\n")