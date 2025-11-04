import os
import sys
import numpy as np
import faiss

from utils.openl3_utils import extract_openl3_embedding
from utils.lyrics_utils import embed_text

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data", "uploads")


AUDIO_INDEX_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "data", "music_index.faiss")
AUDIO_TRACKS_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "data", "track_names.txt")


LYRICS_INDEX_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "data", "lyrics_index.faiss")
LYRICS_TRACKS_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "data", "lyrics_track_names.txt")

TOP_K = 5  # number of top results


def load_index(index_path, names_path):
    if not os.path.exists(index_path) or not os.path.exists(names_path):
        raise FileNotFoundError(f"Index or track names file missing: {index_path}, {names_path}")
    index = faiss.read_index(index_path)
    with open(names_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f]
    return index, names


def query_audio(file_path, index, track_names, top_k=TOP_K):
    emb = extract_openl3_embedding(file_path).astype("float32").reshape(1, -1)
    D, I = index.search(emb, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < len(track_names):
            results.append((track_names[idx], float(dist)))
    return results


def query_lyrics(file_path, index, track_names, top_k=TOP_K):
    if not os.path.exists(file_path):
        print(f"No lyrics file found for {file_path}, skipping lyrics similarity.")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    emb = embed_text(text).astype("float32").reshape(1, -1)
    D, I = index.search(emb, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < len(track_names):
            results.append((track_names[idx], float(dist)))
    return results


def combine_results(audio_results, lyrics_results, alpha=0.5):
    # Normalize distances to similarities (smaller distance = higher similarity)
    audio_sim = {name: 1/(dist+1e-6) for name, dist in audio_results}
    lyrics_sim = {name: 1/(dist+1e-6) for name, dist in lyrics_results}

    all_names = set(audio_sim.keys()) | set(lyrics_sim.keys())
    hybrid_scores = {}
    for name in all_names:
        a = audio_sim.get(name, 0.0)
        l = lyrics_sim.get(name, 0.0)
        hybrid_scores[name] = alpha * a + (1-alpha) * l
    # Sort descending by score
    sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:TOP_K]

def main():
    audio_index, audio_tracks = load_index(AUDIO_INDEX_PATH, AUDIO_TRACKS_PATH)
    lyrics_index, lyrics_tracks = load_index(LYRICS_INDEX_PATH, LYRICS_TRACKS_PATH)

    mp3_files = [f for f in os.listdir(UPLOADS_DIR) if f.lower().endswith(".mp3")]
    if not mp3_files:
        print(f"No .mp3 files found in {UPLOADS_DIR}")
        return

    for mp3 in mp3_files:
        mp3_path = os.path.join(UPLOADS_DIR, mp3)
        lyrics_file = os.path.join(UPLOADS_DIR, os.path.splitext(mp3)[0] + ".txt")

        print(f"\nProcessing {mp3}...")
        audio_results = query_audio(mp3_path, audio_index, audio_tracks)
        lyrics_results = query_lyrics(lyrics_file, lyrics_index, lyrics_tracks)

        hybrid = combine_results(audio_results, lyrics_results)
        print("Top hybrid matches:")
        for i, (name, score) in enumerate(hybrid, 1):
            print(f"{i}. {name} — score {score:.3f}")

if __name__ == "__main__":
    main()
