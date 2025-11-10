
const spotifyPreviewFinder = require('spotify-preview-finder');
const axios = require('axios');
const fs = require('fs');
const path = require('path');


process.env.SPOTIFY_CLIENT_ID = 'ea94426a8cc841778b76c6b9868112df';
process.env.SPOTIFY_CLIENT_SECRET = 'b11df25a3393472da6a5de36ecc83a8e';

const OUTPUT_DIR = path.join(__dirname, '..', 'data', 'uploads');

async function downloadPreview(name, url) {
  const safeName = name.replace(/[^a-z0-9]/gi, '_').toLowerCase();
  const filePath = path.join(OUTPUT_DIR, `${safeName}.mp3`);
  const writer = fs.createWriteStream(filePath);

  const response = await axios({
    url,
    method: 'GET',
    responseType: 'stream'
  });

  response.data.pipe(writer);

  return new Promise((resolve, reject) => {
    writer.on('finish', () => resolve(filePath));
    writer.on('error', reject);
  });
}

async function searchSongs() {
  const songsToSearch = [
    'Radio Bershy',
      'One More Light Linkin Park'
  ];

  if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });

  for (const songQuery of songsToSearch) {
    try {
      const result = await spotifyPreviewFinder(songQuery, 1);
      if (result.success && result.results.length > 0) {
        const song = result.results[0];
        const previewUrl = song.previewUrls[0];

        if (previewUrl) {
          console.log(` Found: ${song.name} - Downloading preview...`);
          const path = await downloadPreview(song.name, previewUrl);
          console.log(` Saved to: ${path}`);
        } else {
          console.log(` No preview found for: ${songQuery}`);
        }
      } else {
        console.log(` No results found for: ${songQuery}`);
      }
    } catch (err) {
      console.error(` Error processing ${songQuery}: ${err.message}`);
    }
  }
}

searchSongs();
