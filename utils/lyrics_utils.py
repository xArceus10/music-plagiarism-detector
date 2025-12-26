import os
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def embed_text(text, normalize=True):
    model = load_model()

    # Encode
    vec = model.encode(text, show_progress_bar=False, convert_to_numpy=True)
    vec = vec.astype("float32")

    # Ensure 1D Vector
    if vec.ndim > 1:
        vec = vec.flatten()

    # Manual Normalization (Cosine Similarity Helper)
    if normalize:
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

    return vec


def embed_file(path, normalize=True):
    if not os.path.exists(path):
        return np.zeros(384).astype("float32")  # Return empty vector if file missing

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return embed_text(text, normalize=normalize)