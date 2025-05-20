import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import os

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="ea94426a8cc841778b76c6b9868112df",
    client_secret="b11df25a3393472da6a5de36ecc83a8e"
))

def fetch_and_download_previews(query, limit=10, output_dir="data/spotify_previews"):
    results = sp.search(q=query, type="track", limit=limit)
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    for item in results['tracks']['items']:
        preview_url = item.get("preview_url")
        if preview_url:
            name = f"{item['name']}_{item['id']}.mp3"
            path = os.path.join(output_dir, name.replace(" ", "_"))
            r = requests.get(preview_url)
            with open(path, 'wb') as f:
                f.write(r.content)
            saved_files.append((name, path))
    return saved_files
