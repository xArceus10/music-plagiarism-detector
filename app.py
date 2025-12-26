import os
import sys
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename

# --- 1. SETUP PATHS ---
# Get the folder where app.py is located
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# UPDATE: Point directly to 'scripts' folder in root
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")

# Add scripts to system path so we can import them
if os.path.exists(SCRIPTS_DIR):
    sys.path.append(SCRIPTS_DIR)
else:
    print(f"❌ ERROR: Could not find folder: {SCRIPTS_DIR}")
    print("   Please ensure 'audio_engine.py' is inside a folder named 'scripts' next to app.py")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(ROOT_DIR, "data", "uploads")

# Create upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- 2. IMPORT ENGINES ---
try:
    import audio_engine
    import lyrics_engine

    # Initialize resources (Lazy load)
    print("🚀 Initializing Engines...")
    audio_engine.init_audio_resources()
    lyrics_engine.init_lyrics_resources()

except ImportError as e:
    print(f"\n❌ IMPORT ERROR: {e}")
    print(f"   Python is looking in: {SCRIPTS_DIR}")
    print("   Make sure the file is named exactly 'audio_engine.py' (no typos).\n")
    # We exit here so you can see the error immediately
    sys.exit(1)


# --- 3. ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        filenames = []
        audio_results = []
        lyrics_results = []

        # --- HANDLE AUDIO ---
        f_audio = request.files.get('audio_file') or request.files.get('file')
        if f_audio and f_audio.filename != '':
            fname = secure_filename(f_audio.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            f_audio.save(path)
            filenames.append(fname)

            # Call Audio Engine
            print(f"🎤 Processing Audio: {fname}")
            audio_results = audio_engine.scan_audio(path)

        # --- HANDLE LYRICS ---
        f_lyrics = request.files.get('lyrics_file')
        if f_lyrics and f_lyrics.filename != '':
            fname = secure_filename(f_lyrics.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            f_lyrics.save(path)
            filenames.append(fname)

            # Call Lyrics Engine
            print(f"📝 Processing Lyrics: {fname}")
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            lyrics_results = lyrics_engine.scan_lyrics(content)

        if not filenames:
            return redirect(request.url)

        return render_template('result.html',
                               filename=" + ".join(filenames),
                               audio_results=audio_results,
                               lyrics_results=lyrics_results)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)