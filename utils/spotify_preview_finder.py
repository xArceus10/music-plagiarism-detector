import requests

def get_preview_url(track_name, artist_name=None, market="US"):
    query = track_name
    if artist_name:
        query += f" {artist_name}"
    headers = {
        "User-Agent": "Mozilla/5.0",
    }
    search_url = f"https://api.spotify.com/v1/search"
    params = {
        "q": query,
        "type": "track",
        "limit": 1,
        "market": market
    }

    # You must have a Spotify token (can be short-lived client credentials token)
    # For now, use your current Spotipy client credentials setup
    from spotipy.oauth2 import SpotifyClientCredentials
    import spotipy

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id="ea94426a8cc841778b76c6b9868112df",
        client_secret="b11df25a3393472da6a5de36ecc83a8e"
    ))

    results = sp.search(q=query, type="track", limit=1)
    if results['tracks']['items']:
        track = results['tracks']['items'][0]
        preview_url = track.get("preview_url")
        if preview_url:
            print(f"🎧 Preview found for '{track['name']}' by {track['artists'][0]['name']}")
            return preview_url
        else:
            print(f"⚠️ No preview available for '{track['name']}'")
            return None
    else:
        print("❌ No track found for that query.")
        return None


from utils.spotify_preview_finder import get_preview_url

preview = get_preview_url("Perfect", "Ed Sheeran")
print("Preview URL:", preview)
