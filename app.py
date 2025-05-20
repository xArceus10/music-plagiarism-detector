from flask import Flask, request
from utils.openl3_utils import extract_openl3_embedding
from utils.vector_index import search_faiss
from utils.human_report import generate_human_report
import faiss
import numpy as np

app = Flask(__name__)

# Load FAISS index and track names
index = faiss.read_index("data/music_index.faiss")
with open("data/track_names.txt", "r") as f:
    names = [line.strip() for line in f.readlines()]

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['audio']
    file_path = "data/uploaded.mp3"
    file.save(file_path)

    user_vector = extract_openl3_embedding(file_path)
    query = np.array(user_vector, dtype='float32').reshape(1, -1)
    D, I = index.search(query, k=3)

    matches = [
        {
            "title": names[i],
            "distance_score": float(D[0][j]),
            "similarity_score": 1 - float(D[0][j]) / np.max(D)
        }
        for j, i in enumerate(I[0])
    ]

    report = generate_human_report(file.filename, matches)
    return report, 200, {'Content-Type': 'text/plain; charset=utf-8'}

if __name__ == '__main__':
    app.run(debug=True)
