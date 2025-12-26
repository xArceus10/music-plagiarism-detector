import os
import sys
import json
import numpy as np
import faiss
import torch
import librosa
import difflib
from scipy.spatial.distance import cdist

# --- PATH SETUP (Essential for App) ---
# Current file is in: .../Music_Plagarism_Detector/scripts/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to get Project Root: .../Music_Plagarism_Detector/
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

# Import utils
try:
    from utils.openl3_utils import extract_openl3_embedding
    from utils.model_def import AudioAdapter
except ImportError:
    print("❌ Audio Engine: Could not import utils. Check sys.path.")

# --- CONFIG (Matches your original) ---
SONGS_DIR = os.path.join(ROOT_DIR, "data", "songs")
AUDIO_INDEX_PATH = os.path.join(ROOT_DIR, "data", "audio_chunked.faiss")
AUDIO_META_PATH = os.path.join(ROOT_DIR, "data", "audio_chunked_meta.json")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "audio_adapter.pth")

# --- GLOBALS (Lazy Loading to prevent crashing) ---
# We load these once so the app doesn't freeze on every upload
LOADED_INDEX = None
LOADED_META = None
LOADED_MODEL = None


def init_audio_resources():
    """
    Loads resources exactly like your original script, but stores them
    in memory so we don't reload them 100 times.
    """
    global LOADED_INDEX, LOADED_META, LOADED_MODEL

    # 1. Load Model
    if os.path.exists(MODEL_PATH):
        try:
            LOADED_MODEL = AudioAdapter()
            LOADED_MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            LOADED_MODEL.eval()
            print("✅ [Audio Engine] Model Loaded")
        except Exception as e:
            print(f"⚠️ [Audio Engine] Model Error: {e}")

    # 2. Load Index & Meta
    if os.path.exists(AUDIO_INDEX_PATH):
        try:
            LOADED_INDEX = faiss.read_index(AUDIO_INDEX_PATH)
            with open(AUDIO_META_PATH, 'r') as f:
                LOADED_META = json.load(f)
            print("✅ [Audio Engine] Index Loaded")
        except Exception as e:
            print(f"⚠️ [Audio Engine] Index Error: {e}")


# --- YOUR ORIGINAL HELPER FUNCTIONS ---

def find_local_file(name, folder):
    # Try exact
    p = os.path.join(folder, name)
    if os.path.exists(p): return p

    # Try clean name
    clean = name.replace("..", "").strip()
    if os.path.exists(folder):
        files = os.listdir(folder)
        # Fuzzy match
        matches = difflib.get_close_matches(clean, files, n=1, cutoff=0.5)
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
        return 1.0 - cost  # EXACT original calculation
    except:
        return 0.0


# --- MAIN LOGIC (Wrapped for App) ---

def scan_audio(audio_path):
    # Ensure resources are loaded
    if LOADED_INDEX is None: init_audio_resources()
    if LOADED_INDEX is None: return []

    print(f"\n🔍 [Audio Engine] Analyzing: {os.path.basename(audio_path)}")

    # 1. Extract & Transform (Your exact logic)
    raw_emb = extract_openl3_embedding(audio_path)
    if raw_emb is None: return []
    if raw_emb.ndim == 1: raw_emb = raw_emb.reshape(1, -1)

    Q = raw_emb
    if LOADED_MODEL:
        with torch.no_grad():
            Q = LOADED_MODEL(torch.from_numpy(Q).float()).numpy()
    else:
        # Your exact fallback normalization
        Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)

    # 2. Search
    D, I = LOADED_INDEX.search(Q, k=1)

    # 3. Vote
    votes = {}
    for dist, idx in zip(D.flatten(), I.flatten()):
        # YOUR ORIGINAL THRESHOLD
        if idx == -1 or dist < 0.65: continue

        name = LOADED_META[idx]['name']
        votes[name] = votes.get(name, 0) + 1

    # Top Audio Candidates
    sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)[:3]

    # 4. Verification & Formatting Results
    final_results = []

    for name, count in sorted_votes:
        # Run Verification (DTW)
        local_path = find_local_file(name, SONGS_DIR)
        dtw_score = 0.0

        if local_path:
            print(f"      Verifying melody with: {name}")
            dtw_score = run_dtw(audio_path, local_path)

        # Append to results in the format App.py needs
        # We multiply by 100 to make it a percentage for the UI
        final_results.append({
            "song": name,
            "score": round(dtw_score * 100, 2)
        })

    return final_results