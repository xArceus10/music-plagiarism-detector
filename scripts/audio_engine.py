import os
import sys
import json
import numpy as np
import faiss
import torch
import librosa
import difflib
from scipy.spatial.distance import cdist

# Setup paths relative to this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(ROOT_DIR)

from utils.openl3_utils import extract_openl3_embedding
from utils.model_def import AudioAdapter

# --- Globals (Loaded Once) ---
AUDIO_INDEX = None
AUDIO_META = None
MODEL = None
SONGS_DIR = os.path.join(ROOT_DIR, "data", "songs")


def init_audio_resources():
    """Loads model and index into memory once."""
    global AUDIO_INDEX, AUDIO_META, MODEL

    # 1. Load FAISS Index
    index_path = os.path.join(ROOT_DIR, "data", "audio_chunked.faiss")
    meta_path = os.path.join(ROOT_DIR, "data", "audio_chunked_meta.json")

    if os.path.exists(index_path) and os.path.exists(meta_path):
        AUDIO_INDEX = faiss.read_index(index_path)
        with open(meta_path, 'r') as f:
            AUDIO_META = json.load(f)
        print("✅ [Audio Engine] Index Loaded")

    # 2. Load Model
    model_path = os.path.join(ROOT_DIR, "models", "audio_adapter.pth")
    if os.path.exists(model_path):
        MODEL = AudioAdapter()
        MODEL.load_state_dict(torch.load(model_path))
        MODEL.eval()
        print("✅ [Audio Engine] Model Loaded")


# --- Helpers ---
def find_local_file(name, folder):
    if not os.path.exists(folder): return None
    p = os.path.join(folder, f"{name}.mp3")
    if os.path.exists(p): return p

    # Fuzzy match
    files = [f for f in os.listdir(folder) if f.endswith('.mp3')]
    matches = difflib.get_close_matches(name, files, n=1, cutoff=0.4)
    if matches:
        return os.path.join(folder, matches[0])
    return None


def run_dtw(path1, path2):
    try:
        y1, sr = librosa.load(path1, sr=22050, mono=True, duration=60)
        y2, _ = librosa.load(path2, sr=22050, mono=True, duration=60)
        c1 = librosa.feature.chroma_cqt(y=y1, sr=sr)
        c2 = librosa.feature.chroma_cqt(y=y2, sr=sr)
        D, wp = librosa.sequence.dtw(C=cdist(c1.T, c2.T, 'cosine'))
        cost = D[-1, -1] / wp.shape[0]

        # Normalize and Boost Score
        similarity = 1.0 - cost
        boosted_score = similarity ** 0.5  # Boost curve
        return boosted_score * 100
    except:
        return 0.0


# --- Main Export Function ---
def scan_audio(audio_path):
    # Ensure resources are loaded
    if AUDIO_INDEX is None: init_audio_resources()
    if AUDIO_INDEX is None: return []  # Failed to load

    print(f"🔍 [Audio Engine] Scanning: {os.path.basename(audio_path)}")

    # 1. Extract & Transform
    raw_emb = extract_openl3_embedding(audio_path)
    if raw_emb is None: return []
    if raw_emb.ndim == 1: raw_emb = raw_emb.reshape(1, -1)

    Q = raw_emb
    if MODEL:
        with torch.no_grad():
            Q = MODEL(torch.from_numpy(Q).float()).numpy()
    else:
        faiss.normalize_L2(Q)

    # 2. Search (Fast)
    D, I = AUDIO_INDEX.search(Q, k=1)

    # 3. Vote
    votes = {}
    for dist, idx in zip(D.flatten(), I.flatten()):
        if idx == -1 or dist < 0.75: continue
        name = AUDIO_META[idx]['name']
        votes[name] = votes.get(name, 0) + 1

    sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)[:3]

    results = []

    # 4. Verify (DTW)
    for name, count in sorted_votes:
        local_path = find_local_file(name, SONGS_DIR)

        score = 50.0  # Fallback
        if local_path:
            score = run_dtw(audio_path, local_path)

        results.append({
            "song": name,
            "score": round(score, 2)
        })

    return results