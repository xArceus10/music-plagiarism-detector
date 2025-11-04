from flask import Flask, render_template, request, jsonify
import os
from scripts.check_hybrid_simf import hybrid_check

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'data', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    audio_file = request.files.get('audio')
    lyrics_file = request.files.get('lyrics')

    if not audio_file:
        return jsonify({'error': 'No audio file uploaded!'}), 400

    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
    audio_file.save(audio_path)

    lyrics_path = None
    if lyrics_file:
        lyrics_path = os.path.join(app.config['UPLOAD_FOLDER'], lyrics_file.filename)
        lyrics_file.save(lyrics_path)

    result = hybrid_check(audio_path, lyrics_path)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
