import os
import re
import sys

# --- Path Setup ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SONGS_DIR = os.path.join(ROOT_DIR, "data", "songs")


def clean_filename(filename):
    name, ext = os.path.splitext(filename)

    # 1. Replace hyphens and underscores with spaces
    name = name.replace('-', ' ').replace('_', ' ')

    # 2. Define junk patterns to remove (case insensitive)
    # Removing website names, "official video", bitrates, brackets, etc.
    junk_patterns = [
        r'savetube\.me', r'savetube', r'ytshorts',
        r'official\s+video', r'official\s+lyric\s+video', r'official\s+audio',
        r'official', r'lyric', r'video', r'audio',
        r'\b128\b', r'\bkbps\b', r'\bhq\b', r'\bhd\b', r'\b4k\b',
        r'www\.[a-z]+\.[a-z]+',  # domains
        r'\[.*?\]',  # content in square brackets
        r'\(.*?\)'  # content in parentheses
    ]

    for pattern in junk_patterns:
        name = re.sub(pattern, '', name, flags=re.IGNORECASE)

    # 3. Clean up extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()

    # 4. Title Case (Make It Look Nice)
    name = name.title()

    return f"{name}{ext}"


def run_cleanup():
    if not os.path.exists(SONGS_DIR):
        print(f"Directory not found: {SONGS_DIR}")
        return

    files = [f for f in os.listdir(SONGS_DIR) if f.lower().endswith(".mp3")]
    print(f"Found {len(files)} files in {SONGS_DIR}...")

    count = 0
    for filename in files:
        new_name = clean_filename(filename)

        # Only rename if the name actually changed
        if new_name != filename:
            old_path = os.path.join(SONGS_DIR, filename)
            new_path = os.path.join(SONGS_DIR, new_name)

            # Handle duplicates (if cleaning makes two files have the same name)
            if os.path.exists(new_path):
                print(f"⚠️ Skipped: '{new_name}' already exists.")
                continue

            try:
                os.rename(old_path, new_path)
                print(f"✨ Renamed: '{filename}' -> '{new_name}'")
                count += 1
            except Exception as e:
                print(f"❌ Error renaming {filename}: {e}")

    print(f"\n✅ cleanup complete. Renamed {count} files.")
    print("❗ IMPORTANT: You must RE-RUN your build_index scripts now!")


if __name__ == "__main__":
    run_cleanup()