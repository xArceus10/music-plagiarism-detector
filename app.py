from flask import Flask, request, render_template
from utils.openl3_utils import extract_openl3_embedding
from utils.vector_index import search_faiss
from utils.human_report import generate_human_report
import numpy as np

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/compare', methods=['POST'])
def compare():
    ref_file = request.files['ref']
    test_file = request.files['test']

    ref_path = "scripts/data/ref_song.mp3"
    test_path = "scripts/data/test_song.mp3"

    ref_file.save(ref_path)
    test_file.save(test_path)

    ref_vector = extract_openl3_embedding(ref_path)
    test_vector = extract_openl3_embedding(test_path)

    ref_vector = np.array(ref_vector, dtype='float32').reshape(1, -1)
    test_vector = np.array(test_vector, dtype='float32').reshape(1, -1)

    dot = np.dot(ref_vector, test_vector.T)[0][0]
    norm = np.linalg.norm(ref_vector) * np.linalg.norm(test_vector)
    similarity = float(dot / norm)

    return {"similarity_score": round(similarity * 100, 2)}, 200

if __name__ == "__main__": app.run(debug=True)