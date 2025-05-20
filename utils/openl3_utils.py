import openl3
import soundfile as sf

def extract_openl3_embedding(file_path):
    audio, sr = sf.read(file_path)
    emb, _ = openl3.get_audio_embedding(audio, sr, content_type="music", embedding_size=512)
    return emb.mean(axis=0)
