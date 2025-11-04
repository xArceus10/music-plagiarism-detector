import sys
import os
import numpy as np
import faiss
import re
import unicodedata

from utils.melody_utils import melody_similarity

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.openl3_utils import extract_openl3_embedding

UPLOADS_DIR = os.path.join(ROOT_DIR, "data", "uploads")
INDEX_PATH = os.path.join(ROOT_DIR, "data", "music_index.faiss")
TRACK_LIST_PATH = os.path.join(ROOT_DIR, "data", "track_names.txt")
EMBEDDINGS_PATH = os.path.join(ROOT_DIR, "data", "music_embeddings.npy")

TOP_K = 5
SIMILARITY_THRESHOLD = 0.85


def cosine_similarity(v1, v2):
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    return np.dot(v1_norm, v2_norm)


def safe_filename(name: str) -> str:
    # Normalize Unicode (NFKD)
    name = unicodedata.normalize("NFKD", name)

    # Remove accents
    name = "".join(c for c in name if not unicodedata.combining(c))

    # Remove invalid Windows filename characters
    name = re.sub(r'[<>:"/\\|?*]+', "", name)

    # Collapse multiple spaces into one
    name = re.sub(r"\s+", " ", name)

    # Trim spaces
    name = name.strip()

    return name


def check_song(path_to_upload):
    if not os.path.exists(INDEX_PATH):
        print("Index not found. Please run build_index.py first.")
        return

    print(f"\nAnalyzing: {path_to_upload}")
    query_emb = extract_openl3_embedding(path_to_upload).astype("float32")

    index = faiss.read_index(INDEX_PATH)
    all_embeddings = np.load(EMBEDDINGS_PATH)

    with open(TRACK_LIST_PATH, "r", encoding="utf-8") as f:
        track_names = [line.strip() for line in f.readlines()]

    distances, indices = index.search(np.array([query_emb]), k=TOP_K)
    candidates = indices[0]

    print("\n🎼 Similar Tracks Found:\n")

    seen = set()
    for i, idx in enumerate(candidates):
        name = track_names[idx]
        if name in seen:
            continue  # skip duplicates
        seen.add(name)

        original_emb = all_embeddings[idx]
        cos_sim = cosine_similarity(query_emb, original_emb)

        # Add .mp3 extension so it matches downloader filenames
        candidate_file = f"{safe_filename(name)}.mp3"
        candidate_path = os.path.join("data", "spotify_previews", candidate_file)

        if not os.path.exists(candidate_path):
            print(f"   (Preview not found for {name}, expected {candidate_file})")
            mel_sim = 0.0
        else:
            mel_sim = melody_similarity(path_to_upload, candidate_path)

        print(f"{i+1}. {name}")
        print(f"    🔹 Cosine similarity (OpenL3): {cos_sim:.3f}")
        print(f"    🔹 Melody similarity (DTW chroma): {mel_sim:.3f}")

        if cos_sim > SIMILARITY_THRESHOLD or mel_sim > 0.6:
            print(" ️Possible plagiarism")
        print()

    print(" Done.")


if __name__ == "__main__":
    # Case 1: user passed a file
    if len(sys.argv) == 2:
        audio_path = sys.argv[1]

    # Case 2: auto-pick first MP3 in uploads folder
    else:
        mp3_files = [f for f in os.listdir(UPLOADS_DIR) if f.lower().endswith(".mp3")]
        if not mp3_files:
            print(" No MP3 files found in uploads folder.")
            sys.exit(1)
        audio_path = os.path.join(UPLOADS_DIR, mp3_files[0])
        print(f"⚡ No file passed, using first upload: {audio_path}")

    check_song(audio_path)
