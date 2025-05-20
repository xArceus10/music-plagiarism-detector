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
    print(f"Search returned {len(results['tracks']['items'])} tracks for query '{query}'")
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    for item in results['tracks']['items']:
        preview_url = item.get("preview_url")
        print(f"Track: {item['name']} by {item['artists'][0]['name']}")
        if preview_url:
            print(f"  Preview URL: {preview_url}")
            name = f"{item['name']}_{item['id']}.mp3"
            path = os.path.join(output_dir, name.replace(" ", "_"))
            r = requests.get(preview_url)
            if r.status_code == 200 and len(r.content) > 1000:
                with open(path, 'wb') as f:
                    f.write(r.content)
                saved_files.append((name, path))
            else:
                print(f"  Failed to download preview or file too small")
        else:
            print("  No preview available")
    return saved_files
if __name__ == "__main__":
    previews = fetch_and_download_previews("Ed Sheeran", limit=5)
    print(f"Downloaded {len(previews)} previews:")
    for name, path in previews:
        print(f" - {name} saved to {path}")

