import os
import sys
import json
import numpy as np
import faiss
import torch
import librosa
import difflib
from scipy.spatial.distance import cdist

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.openl3_utils import extract_openl3_embedding
from utils.lyrics_utils import embed_text
from utils.model_def import AudioAdapter

# --- Config ---
UPLOAD_DIR = os.path.join(ROOT_DIR, "data", "uploads")
SONGS_DIR = os.path.join(ROOT_DIR, "data", "songs")
AUDIO_INDEX = os.path.join(ROOT_DIR, "data", "audio_chunked.faiss")
AUDIO_META = os.path.join(ROOT_DIR, "data", "audio_chunked_meta.json")
LYRICS_INDEX = os.path.join(ROOT_DIR, "data", "lyrics_index.faiss")
LYRICS_NAMES = os.path.join(ROOT_DIR, "data", "lyrics_track_names.txt")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "audio_adapter.pth")


# --- Helper: Find File ---
def find_local_file(name, folder):
    # Try exact
    p = os.path.join(folder, name)
    if os.path.exists(p): return p

    # Try clean name
    clean = name.replace("..", "").strip()
    files = os.listdir(folder)

    # Fuzzy match
    matches = difflib.get_close_matches(clean, files, n=1, cutoff=0.5)
    if matches:
        return os.path.join(folder, matches[0])
    return None


# --- Helper: DTW ---
def run_dtw(path1, path2):
    try:
        y1, sr = librosa.load(path1, sr=22050, mono=True, duration=60)
        y2, _ = librosa.load(path2, sr=22050, mono=True, duration=60)
        c1 = librosa.feature.chroma_cqt(y=y1, sr=sr)
        c2 = librosa.feature.chroma_cqt(y=y2, sr=sr)
        D, wp = librosa.sequence.dtw(C=cdist(c1.T, c2.T, 'cosine'))
        cost = D[-1, -1] / wp.shape[0]
        return 1.0 - cost  # Convert cost to similarity
    except:
        return 0.0


# --- Main Logic ---
def check_plagiarism(audio_path, lyrics_text=None):
    print(f"\n🔍 Analyzing: {os.path.basename(audio_path)}")
    results = {}

    # 1. LOAD MODEL
    model = None
    if os.path.exists(MODEL_PATH):
        model = AudioAdapter()
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()

    # 2. AUDIO SEARCH (Chunking)
    if os.path.exists(AUDIO_INDEX):
        print("   -> Audio Scan...")
        index = faiss.read_index(AUDIO_INDEX)
        with open(AUDIO_META, 'r') as f:
            meta = json.load(f)

        # Extract & Transform
        raw_emb = extract_openl3_embedding(audio_path)
        if raw_emb.ndim == 1: raw_emb = raw_emb.reshape(1, -1)

        # Chunking (Simulated)
        # For simplicity in search, we query every 10th frame or average chunks
        # Here we just use the raw frames as queries

        Q = raw_emb
        if model:
            with torch.no_grad():
                Q = model(torch.from_numpy(Q).float()).numpy()
        else:
            Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)

        # Search
        D, I = index.search(Q, k=1)

        # Vote
        votes = {}
        for dist, idx in zip(D.flatten(), I.flatten()):
            if idx == -1 or dist < 0.65: continue
            name = meta[idx]['name']
            votes[name] = votes.get(name, 0) + 1

        # Top Audio Candidate
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)[:3]
        results['audio_matches'] = []

        for name, count in sorted_votes:
            # Run Verification (DTW)
            local_path = find_local_file(name, SONGS_DIR)
            dtw_score = 0.0
            if local_path:
                print(f"      Verifying melody with: {name}")
                dtw_score = run_dtw(audio_path, local_path)

            results['audio_matches'].append({
                "song": name,
                "chunk_votes": count,
                "melody_dtw": round(dtw_score, 3)
            })

    # 3. LYRICS SEARCH
    if lyrics_text and os.path.exists(LYRICS_INDEX):
        print("   -> Lyrics Scan...")
        l_index = faiss.read_index(LYRICS_INDEX)
        with open(LYRICS_NAMES, 'r') as f:
            l_names = [x.strip() for x in f]

        l_vec = embed_text(lyrics_text).astype('float32').reshape(1, -1)
        l_vec = l_vec / (np.linalg.norm(l_vec) + 1e-12)

        D, I = l_index.search(l_vec, k=3)
        results['lyrics_matches'] = []
        for d, i in zip(D[0], I[0]):
            if i < len(l_names):
                results['lyrics_matches'].append({
                    "song": l_names[i],
                    "sim": float(d)
                })

    # 4. REPORT
    print("\n" + "=" * 30)
    print("FINAL REPORT")
    print("=" * 30)

    if results.get('audio_matches'):
        top = results['audio_matches'][0]
        print(f"🔊 Top Audio Match: {top['song']}")
        print(f"   Confidence (DTW): {top['melody_dtw'] * 100:.1f}%")
        if top['melody_dtw'] > 0.6:
            print("   🚨 DECISION: Potential Audio Plagiarism Detected")
        else:
            print("   ✅ DECISION: Audio structure is different.")
    else:
        print("🔊 No significant audio matches.")

    if results.get('lyrics_matches'):
        top_l = results['lyrics_matches'][0]
        print(f"\n📝 Top Lyrical Match: {top_l['song']}")
        print(f"   Similarity: {top_l['sim']:.2f}")

    print("=" * 30)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 5_main_check.py <filename.mp3> [optional_lyrics.txt]")
        # Auto-pick for testing
        files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".mp3")]
        if files:
            target = os.path.join(UPLOAD_DIR, files[0])
            check_plagiarism(target)
    else:
        # Parse args
        mp3 = sys.argv[1]
        txt = sys.argv[2] if len(sys.argv) > 2 else None

        # Read text content if file provided
        lyr_content = None
        if txt and os.path.exists(txt):
            with open(txt, 'r') as f: lyr_content = f.read()

        check_plagiarism(mp3, lyr_content)