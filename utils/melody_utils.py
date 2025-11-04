import librosa
import numpy as np
from dtw import accelerated_dtw


def extract_chroma(path, sr=22050):
    """
    Extracts chroma (pitch class representation) from audio.
    """
    y, sr = librosa.load(path, sr=sr)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    return chroma.T  # shape: (time, 12)


def melody_similarity(path1, path2):
    """
    Compares two audio files based on chroma features + DTW.
    Returns similarity score in [0,1].
    """
    c1 = extract_chroma(path1)
    c2 = extract_chroma(path2)

    dist, _, _, _ = accelerated_dtw(c1, c2, dist='euclidean')
    sim = 1 / (1 + dist)  # normalize
    return float(sim)
