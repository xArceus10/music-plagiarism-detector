from scripts.check_audio_sim import melody_similarity

mel = melody_similarity("data/uploads/Eagles - Hotel California.mp3", "data/spotify_previews/Another Love - Tom Odell.mp3")
print("Melody similarity =", mel)
