import os
import re
import sys
import requests
from bs4 import BeautifulSoup

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

GENIUS_API_BASE = "https://api.genius.com"
GENIUS_ACCESS_TOKEN = "f_mGJhCfv9a_GXF4ZnQlh6aEis9ORQ3eTr70xhkyE3aOijlVUM-lrBLt0HP-A1Fy"  # Your token
headers = {"Authorization": f"Bearer {GENIUS_ACCESS_TOKEN}"}

# --- Base Paths (relative to this script) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data"))
PREVIEWS_DIR = os.path.join(BASE_DIR, "songs")
LYRICS_DIR = os.path.join(BASE_DIR, "lyrics")


def clean_lyrics(raw_text):
    # (Your clean_lyrics function is perfect, no changes)
    lines = raw_text.splitlines()
    clean_lines = []
    started = False
    for line in lines:
        line = line.strip()
        if not line: continue
        if any(skip in line.lower() for skip in
               ["contributors", "translations", "read more", "tracklist", "about"]): continue
        if re.match(r"^[\u4e00-\u9fff\u3040-\u30ff\u0600-\u06ff]+$", line): continue
        if line.startswith("[") and line.endswith("]"):
            started = True
            clean_lines.append(line)
            continue
        if started:
            clean_lines.append(line)
    return "\n".join(clean_lines)


def fetch_lyrics(song_title, artist_name=""):

    print(f"Searching Genius for '{song_title} {artist_name}'...")
    search_url = f"{GENIUS_API_BASE}/search"
    params = {"q": f"{song_title} {artist_name}".strip()}
    response = requests.get(search_url, params=params, headers=headers)
    if response.status_code != 200:
        print(f"Error: Genius API request failed ({response.status_code})")
        return None
    try:
        data = response.json()
        hits = data.get("response", {}).get("hits", [])
        if not hits:
            print("No results found on Genius.")
            return None
        hit = hits[0]["result"]
        song_url = hit["url"]
        print(f"Found: {hit['title']} by {hit['primary_artist']['name']}")
        page = requests.get(song_url)
        soup = BeautifulSoup(page.text, "html.parser")
        lyrics_divs = soup.find_all("div", class_=re.compile("Lyrics__Container|lyrics"))
        if not lyrics_divs:
            print("No lyrics found on page.")
            return None
        raw_text = "\n".join([div.get_text(separator="\n") for div in lyrics_divs])
        return clean_lyrics(raw_text)
    except Exception as e:
        print(f"Error parsing Genius response: {e}")
        return None


# --- NEW REUSABLE FUNCTION ---
def parse_filename(filename):
    """Parses 'Artist - Title.mp3' into (Title, Artist)."""
    clean_name = os.path.splitext(filename)[0].replace("_", " ")
    parts = clean_name.split("-")

    if len(parts) >= 2:
        song_title = parts[0].strip()
        artist_name = " ".join(parts[1:]).strip()
    else:
        song_title = clean_name.strip()
        artist_name = ""
    return song_title, artist_name


def fetch_lyrics_from_previews():

    os.makedirs(LYRICS_DIR, exist_ok=True)
    for file in os.listdir(PREVIEWS_DIR):
        if not file.endswith(".mp3"):
            continue

        # Use the new helper function
        song_title, artist_name = parse_filename(file)

        print(f"\nFetching lyrics for: {song_title} by {artist_name}")

        lyrics = fetch_lyrics(song_title, artist_name)
        if lyrics:
            txt_path = os.path.join(LYRICS_DIR, f"{os.path.splitext(file)[0]}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(lyrics)
            print(f"Lyrics saved to {txt_path}")
        else:
            print(f"No lyrics found for {file}")


if __name__ == "__main__":
    fetch_lyrics_from_previews()