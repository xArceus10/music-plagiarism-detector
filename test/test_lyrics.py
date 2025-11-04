import os
import re
import requests
from bs4 import BeautifulSoup

GENIUS_API_BASE = "https://api.genius.com"
GENIUS_ACCESS_TOKEN = "f_mGJhCfv9a_GXF4ZnQlh6aEis9ORQ3eTr70xhkyE3aOijlVUM-lrBLt0HP-A1Fy"  #

headers = {"Authorization": f"Bearer {GENIUS_ACCESS_TOKEN}"}

# Directory where previews are stored
PREVIEWS_DIR = r"C:\Users\Jai\PycharmProjects\Music_Plagarism_Detector\data\spotify_previews"

def clean_lyrics(raw_text):
    """Remove translations, metadata, and only keep song lyrics."""
    lines = raw_text.splitlines()
    clean_lines = []

    started = False
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip metadata/translation/boilerplate
        if any(skip in line.lower() for skip in [
            "contributors", "translations", "read more", "tracklist", "about"
        ]):
            continue

        # Skip obvious translation / foreign language lines
        if re.match(r"^[\u4e00-\u9fff\u3040-\u30ff\u0600-\u06ff]+$", line):
            continue

        # Start keeping after first section header like [Verse] / [Chorus]
        if line.startswith("[") and line.endswith("]"):
            started = True
            clean_lines.append(line)
            continue

        if started:
            clean_lines.append(line)

    return "\n".join(clean_lines)

def fetch_lyrics(song_title, artist_name=""):
    """Fetch lyrics from Genius API and scrape the page."""
    print(f"Searching Genius for '{song_title} {artist_name}'...")

    search_url = f"{GENIUS_API_BASE}/search"
    params = {"q": f"{song_title} {artist_name}".strip()}
    response = requests.get(search_url, params=params, headers=headers)

    if response.status_code != 200:
        print(f"Error: Genius API request failed ({response.status_code})")
        return None

    data = response.json()
    hits = data.get("response", {}).get("hits", [])
    if not hits:
        print("No results found.")
        return None

    # Pick first hit
    hit = hits[0]["result"]
    song_url = hit["url"]
    print(f"Found: {hit['title']} by {hit['primary_artist']['name']}")
    print(f"URL: {song_url}")

    # Scrape lyrics page
    page = requests.get(song_url)
    soup = BeautifulSoup(page.text, "html.parser")
    lyrics_divs = soup.find_all("div", class_=re.compile("Lyrics__Container|lyrics"))

    if not lyrics_divs:
        print("No lyrics found on page.")
        return None

    raw_text = "\n".join([div.get_text(separator="\n") for div in lyrics_divs])
    return clean_lyrics(raw_text)

def fetch_lyrics_from_previews():
    """Go through preview MP3s and fetch matching lyrics."""
    for file in os.listdir(PREVIEWS_DIR):
        if not file.endswith(".mp3"):
            continue

        # Try to extract title + artist from filename
        song_query = os.path.splitext(file)[0].replace("_", " ")
        parts = song_query.split("-")

        if len(parts) >= 2:
            song_title = parts[0].strip()
            artist_name = " ".join(parts[1:]).strip()
        else:
            song_title = song_query.strip()
            artist_name = ""

        print(f"\nFetching lyrics for: {song_title} by {artist_name}")

        lyrics = fetch_lyrics(song_title, artist_name)
        if lyrics:
            txt_path = os.path.join(PREVIEWS_DIR, f"{os.path.splitext(file)[0]}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(lyrics)
            print(f"Lyrics saved to {txt_path}")
        else:
            print(f"No lyrics found for {file}")

if __name__ == "__main__":
    fetch_lyrics_from_previews()
