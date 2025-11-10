import os
import sys

# --- Path Setup ---
# Ensures we can import from other scripts in the 'scripts' folder
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Import the functions we need from your other script
try:
    from scripts.genius_lyrics_fetcher import fetch_lyrics, parse_filename
except ImportError:
    print("Error: Could not find 'scripts/fetch_genius.py'.")
    print("Please make sure that file exists and is in the 'scripts' folder.")
    sys.exit(1)

UPLOADS_DIR = os.path.join(ROOT_DIR, "data", "uploads")


def get_lyrics_for_upload():
    """
    Finds the first MP3 in data/uploads, fetches its lyrics,
    and saves them to a .txt file in the same folder.
    """
    if not os.path.exists(UPLOADS_DIR):
        print(f"Error: Uploads directory not found at {UPLOADS_DIR}")
        return

    # Find the first .mp3 file
    mp3_file = None
    for f in os.listdir(UPLOADS_DIR):
        if f.lower().endswith(".mp3"):
            mp3_file = f
            break

    if not mp3_file:
        print(f"No .mp3 files found in {UPLOADS_DIR}.")
        print("Please add an MP3 file to the uploads folder first.")
        return

    print(f"Found audio file: {mp3_file}")

    # 1. Parse the filename
    song_title, artist_name = parse_filename(mp3_file)

    # 2. Fetch the lyrics
    lyrics_text = fetch_lyrics(song_title, artist_name)

    if lyrics_text:
        # 3. Save the lyrics to the uploads folder
        base_name = os.path.splitext(mp3_file)[0]
        txt_filename = f"{base_name}.txt"
        save_path = os.path.join(UPLOADS_DIR, txt_filename)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(lyrics_text)

        print(f"\n✅ Success! Lyrics saved to:\n{save_path}")
    else:
        print(f"\n❌ Failed to find lyrics for '{song_title}'.")


if __name__ == "__main__":
    get_lyrics_for_upload()