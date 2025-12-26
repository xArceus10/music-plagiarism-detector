import openl3
import soundfile as sf
import numpy as np
import librosa


def extract_openl3_embedding(file_path):
    """
    Returns:
        numpy array of shape (Time_Steps, 512)
    """
    try:
        # Load audio using librosa to ensure consistent sample rate and mono
        # (soundfile can sometimes fail on 24-bit headers or varying channels)
        audio, sr = librosa.load(file_path, sr=48000, mono=True)

        # Get embeddings (hop_size=1.0 means 1 vector per second)
        emb, ts = openl3.get_audio_embedding(
            audio,
            sr,
            content_type="music",
            embedding_size=512,
            hop_size=1.0
        )

        return emb  # Returns shape (N, 512) - DO NOT MEAN HERE!

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Return empty array to prevent crashes
        return np.empty((0, 512))