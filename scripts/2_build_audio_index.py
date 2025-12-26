import os
import sys
import json
import numpy as np
import faiss
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.openl3_utils import extract_openl3_embedding
from utils.model_def import AudioAdapter

# Config
SONGS_DIR = os.path.join(ROOT_DIR, "data", "songs")
INDEX_PATH = os.path.join(ROOT_DIR, "data", "audio_chunked.faiss")
METADATA_PATH = os.path.join(ROOT_DIR, "data", "audio_chunked_meta.json")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "audio_adapter.pth")

CHUNK_SIZE = 10.0
HOP_SIZE = 5.0


def load_ai_model():
    if os.path.exists(MODEL_PATH):
        print("🔹 Loading trained AI Adapter...")
        model = AudioAdapter()
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        return model
    else:
        print("🔸 No AI model found. Using raw OpenL3 vectors (Lower accuracy).")
        return None


def process_file(filepath, model):
    # 1. Extract Raw OpenL3 (Shape: T x 512)
    try:
        full_emb = extract_openl3_embedding(filepath)
    except:
        return [], []

    if full_emb.ndim == 1: full_emb = full_emb.reshape(1, -1)

    vectors = []
    metadata = []
    filename = os.path.basename(filepath)
    num_seconds = full_emb.shape[0]

    # 2. Chunking Loop
    for start in range(0, num_seconds, int(HOP_SIZE)):
        end = start + int(CHUNK_SIZE)
        if end > num_seconds and start > 0: break

        # Average the chunk
        chunk_raw = np.mean(full_emb[start:end], axis=0)

        # 3. Apply AI Model (Dimension Reduction 512 -> 128)
        if model:
            with torch.no_grad():
                tensor_in = torch.from_numpy(chunk_raw).float().unsqueeze(0)  # Add batch dim
                tensor_out = model(tensor_in)
                final_vec = tensor_out.numpy().flatten()
        else:
            # Fallback: Just normalize the raw 512 vector
            final_vec = chunk_raw / (np.linalg.norm(chunk_raw) + 1e-12)

        vectors.append(final_vec)
        metadata.append({"name": filename, "time": f"{start}-{end}s"})

    return vectors, metadata


def build():
    model = load_ai_model()
    all_vecs = []
    all_meta = []

    files = [f for f in os.listdir(SONGS_DIR) if f.endswith(".mp3")]
    print(f"Found {len(files)} songs.")

    for i, f in enumerate(files):
        print(f"Processing [{i + 1}/{len(files)}] {f}...")
        v, m = process_file(os.path.join(SONGS_DIR, f), model)
        all_vecs.extend(v)
        all_meta.extend(m)

    if not all_vecs: return

    # 4. Create Index
    X = np.array(all_vecs).astype('float32')
    d = X.shape[1]  # Will be 128 if model used, 512 if not
    print(f"Building Index with vector dimension: {d}")

    index = faiss.IndexFlatIP(d)
    index.add(X)

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, 'w') as f:
        json.dump(all_meta, f)

    print("✅ Indexing Complete.")


if __name__ == "__main__":
    build()