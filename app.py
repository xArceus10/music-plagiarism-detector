import os
import json
import sys  # Make sure sys is imported
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import subprocess

# --- Configuration ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(ROOT_DIR, "data", "uploads")
# --- This is the line you change to switch scripts ---
HYBRID_SCRIPT_PATH = os.path.join(ROOT_DIR, "scripts", "check_hybrid_fast.py")

# --- Import Genius Lyric Fetcher ---
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

try:
    # Assumes your fetcher script is named 'fetch_genius.py'
    from fetch_genius import fetch_lyrics, parse_filename

    print("Successfully imported Genius lyric fetcher.")
except ImportError:
    print("WARNING: Could not import fetch_genius.py. Auto-fetching disabled.")
    fetch_lyrics = None
    parse_filename = None  # Make sure to null this too

# Ensure the uploads directory exists
os.makedirs(UPLOADS_DIR, exist_ok=True)

app = Flask(__name__)  # This automatically finds the 'templates' folder
app.config['UPLOAD_FOLDER'] = UPLOADS_DIR

# --- A simple "database" to store results ---
results_db = {}


@app.route('/')
def index():
    """Serves the main upload page."""
    # This now correctly uses your templates/index.html file
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handles file uploads and runs the analysis."""
    if 'audio_file' not in request.files:
        return "No audio file part", 400

    audio_file = request.files['audio_file']
    lyrics_file = request.files.get('lyrics_file')

    if audio_file.filename == '':
        return "No selected audio file", 400

    # --- Clear old uploads ---
    print("Cleaning uploads directory...")
    for f in os.listdir(UPLOADS_DIR):
        try:
            os.remove(os.path.join(UPLOADS_DIR, f))
        except Exception as e:
            print(f"Warning: could not remove file {f}: {e}")

    # --- Save audio file ---
    audio_filename = secure_filename(audio_file.filename)
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
    audio_file.save(audio_path)
    print(f"Saved audio: {audio_filename}")

    # --- Handle Lyrics (with Genius fetch) ---
    lyrics_path = None
    if lyrics_file and lyrics_file.filename != '':
        # Case 1: User uploaded a lyrics file
        print("User provided lyrics file.")
        lyrics_filename = secure_filename(lyrics_file.filename)
        lyrics_path = os.path.join(app.config['UPLOAD_FOLDER'], lyrics_filename)
        lyrics_file.save(lyrics_path)

    elif fetch_lyrics and parse_filename:
        # Case 2: No lyrics file, attempt to fetch
        print("No lyrics file provided. Attempting to fetch from Genius...")
        song_title, artist_name = parse_filename(audio_filename)

        lyrics_text = fetch_lyrics(song_title, artist_name)

        if lyrics_text:
            # Save the fetched lyrics to a file
            base_name = os.path.splitext(audio_filename)[0]
            lyrics_filename = f"{base_name}.txt"  # Match the mp3 name
            lyrics_path = os.path.join(app.config['UPLOAD_FOLDER'], lyrics_filename)
            with open(lyrics_path, "w", encoding="utf-8") as f:
                f.write(lyrics_text)
            print(f"Successfully fetched and saved lyrics to {lyrics_path}")
        else:
            print("Failed to fetch lyrics from Genius.")
    else:
        # Case 3: No lyrics file and fetcher is not working
        print("No lyrics file and fetcher is disabled. Proceeding without lyrics.")

    # --- Run the hybrid script using subprocess ---
    print(f"Running analysis script: {HYBRID_SCRIPT_PATH}")

    python_exe = sys.executable

    try:
        process = subprocess.run(
            [python_exe, HYBRID_SCRIPT_PATH],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )

        # Find the JSON output at the very end of the script's stdout
        json_output = None
        for line in reversed(process.stdout.splitlines()):
            if line.strip().startswith('{'):
                json_output = line.strip()
                break

        if json_output:
            result_data = json.loads(json_output)
            # Store result in our 'db' using a simple ID
            result_id = 'last_result'
            results_db[result_id] = result_data
            return redirect(url_for('show_result', result_id=result_id))
        else:
            # Show the raw log if JSON parsing failed
            return f"Error: Could not find JSON output.<br><pre>{process.stdout}</pre><br><pre>{process.stderr}</pre>"

    except subprocess.CalledProcessError as e:
        return f"Script failed:<br><pre>{e.stdout}</pre><br><pre>{e.stderr}</pre>"
    except Exception as e:
        return f"An error occurred: {str(e)}"


@app.route('/result/<result_id>')
def show_result(result_id):
    """Displays the results page."""
    result = results_db.get(result_id)
    if not result:
        return "Result not found", 404

    # Determine decision color
    decision = result.get('hybrid_decision', '').lower()
    if '🔴' in decision or 'strong' in decision:
        decision_color = 'red'
    elif '🟠' in decision or 'high' in decision:
        decision_color = 'orange'
    elif '🟡' in decision or 'moderate' in decision:
        decision_color = 'yellow'
    else:
        decision_color = 'green'

    # Pass raw JSON for debugging
    raw_json = json.dumps(result, indent=4)

    # This now correctly uses your templates/result.html file
    return render_template('result.html',
                           result=result,
                           raw_json=raw_json,
                           decision_color=decision_color)



if __name__ == '__main__':
    print("\n---")
    print("Starting Flask server...")
    print(f"Using hybrid script: {HYBRID_SCRIPT_PATH}")
    print("Go to http://127.0.0.1:5000 in your web browser.")
    print("---")
    app.run(debug=True, port=5000)