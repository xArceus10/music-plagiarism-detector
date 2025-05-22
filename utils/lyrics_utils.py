from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')  # Light and fast

def extract_lyrics_embedding(lyrics_text):
    embedding = model.encode([lyrics_text], convert_to_tensor=False)
    return np.array(embedding).astype('float32')
