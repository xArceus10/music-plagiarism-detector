import os

import sys

import numpy as np

import faiss





ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(ROOT_DIR)



from utils.lyrics_utils import embed_text





UPLOADS_DIR = os.path.join(ROOT_DIR, "data", "uploads")  # Folder with .txt files

INDEX_PATH = os.path.join(ROOT_DIR, "data", "lyrics_index.faiss")

NAMES_PATH = os.path.join(ROOT_DIR, "data", "lyrics_track_names.txt")

TOP_K = 5





def load_index():

    if not os.path.exists(INDEX_PATH) or not os.path.exists(NAMES_PATH):

        raise FileNotFoundError("Lyrics index or names file missing. Run build_lyrics_index.py first.")

    index = faiss.read_index(INDEX_PATH)

    with open(NAMES_PATH, "r", encoding="utf-8") as f:

        names = [line.strip() for line in f]

    return index, names





def query_text(text, top_k=TOP_K):

    index, names = load_index()

    q = embed_text(text).astype('float32').reshape(1, -1)

    D, I = index.search(q, k=top_k)

    return [(names[idx], float(dist)) for dist, idx in zip(D[0], I[0])]



def query_file(path, top_k=TOP_K):

    with open(path, "r", encoding="utf-8") as f:

        text = f.read()

    return query_text(text, top_k=top_k)





if __name__ == "__main__":

    if not os.path.exists(UPLOADS_DIR):

        print(f"No uploads found in {UPLOADS_DIR}")

        sys.exit(1)



    txt_files = [f for f in os.listdir(UPLOADS_DIR) if f.lower().endswith(".txt")]

    if not txt_files:

        print(f"No .txt files found in {UPLOADS_DIR}")

        sys.exit(1)



    for txt_file in txt_files:

        txt_path = os.path.join(UPLOADS_DIR, txt_file)

        print(f"\nProcessing {txt_file}...")

        results = query_file(txt_path)

        print("Top matches (lyrics):")

        for i, (name, sim) in enumerate(results, 1):

            print(f"{i}. {name} — similarity {sim:.3f}")



