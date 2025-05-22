import faiss
import numpy as np
from utils.lyrics_utils import extract_lyrics_embedding
import os

lyrics_folder = "data/lyrics"
track_names = []
embeddings = []

for filename in os.listdir(lyrics_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(lyrics_folder, filename), "r", encoding="utf-8") as f:
            lyrics = f.read()
        emb = extract_lyrics_embedding(lyrics)
        embeddings.append(emb[0])
        track_names.append(filename.replace(".txt", ""))

vectors = np.vstack(embeddings).astype('float32')
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)

faiss.write_index(index, "data/lyrics_index.faiss")
with open("data/track_names.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(track_names))
