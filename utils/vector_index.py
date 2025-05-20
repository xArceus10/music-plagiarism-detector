import faiss
import numpy as np
from utils.openl3_utils import extract_openl3_embedding

def build_faiss_index(audio_files):
    dimension = 512
    index = faiss.IndexFlatL2(dimension)
    vectors = []
    names = []
    for name, path in audio_files:
        emb = extract_openl3_embedding(path)
        vectors.append(emb)
        names.append(name)
    vectors_np = np.vstack(vectors).astype('float32')
    index.add(vectors_np)
    return index, names, vectors_np

def search_faiss(query_vector, index, names, k=3):
    query_vector = np.array(query_vector, dtype='float32').reshape(1, -1)
    D, I = index.search(query_vector, k)
    results = [(names[i], D[0][j]) for j, i in enumerate(I[0])]
    return results
