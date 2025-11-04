const spotifyPreviewFinder = require('spotify-preview-finder');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const readline = require('readline');

process.env.SPOTIFY_CLIENT_ID = 'ea94426a8cc841778b76c6b9868112df';
process.env.SPOTIFY_CLIENT_SECRET = 'b11df25a3393472da6a5de36ecc83a8e';

const OUTPUT_DIR = path.join(__dirname, '..', 'data', 'spotify_previews');

async function downloadPreview(name, url) {

  const safeName = name
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "") // remove accents
    .replace(/[<>:"/\\|?*]+/g, "")   // remove invalid Windows characters
    .replace(/\s+/g, " ")            // collapse multiple spaces
    .trim();

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

async function getAccessToken() {
  const clientId = process.env.SPOTIFY_CLIENT_ID;
  const clientSecret = process.env.SPOTIFY_CLIENT_SECRET;

  if (!clientId || !clientSecret) {
    throw new Error('SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET not set');
  }

  const tokenResp = await axios.post(
    'https://accounts.spotify.com/api/token',
    new URLSearchParams({ grant_type: 'client_credentials' }).toString(),
    {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        Authorization: 'Basic ' + Buffer.from(`${clientId}:${clientSecret}`).toString('base64'),
      },
      timeout: 10000
    }
  );

  return tokenResp.data.access_token;
}

async function fetchPlaylistAndDownloadPreviews(playlistId) {
  if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });

  let token;
  try {
    token = await getAccessToken();
  } catch (err) {
    console.error(' Failed to get Spotify token:', err.message || err);
    return;
  }

  let nextUrl = `https://api.spotify.com/v1/playlists/${playlistId}/tracks?limit=100`;
  const metadata = [];

  while (nextUrl) {
    let resp;
    try {
      resp = await axios.get(nextUrl, {
        headers: { Authorization: `Bearer ${token}` },
        timeout: 15000
      });
    } catch (err) {
      console.error(' Spotify API error fetching playlist page:', err.message || err);
      if (err.response) {
        console.error('Status:', err.response.status, 'Body:', err.response.data);
      }
      break;
    }

    const pageData = resp.data;
    if (!pageData || !Array.isArray(pageData.items)) {
      console.error(' Unexpected playlist response shape, aborting.');
      console.error(JSON.stringify(pageData, null, 2));
      break;
    }

    for (const item of pageData.items) {
      const track = item && item.track;
      if (!track) {
        console.log(' Skipping an empty/removed track entry');
        continue;
      }

      const query = `${track.name} ${track.artists.map(a => a.name).join(' ')}`.trim();
      console.log(`🔎 Searching previews for: ${query}`);

      try {
        const result = await spotifyPreviewFinder(query, 1);

        if (result && result.success && Array.isArray(result.results) && result.results.length > 0) {
          const song = result.results[0];
          const previewUrl = Array.isArray(song.previewUrls) ? song.previewUrls[0] : song.previewUrl || null;

          if (previewUrl) {
            console.log(` Found preview for: ${song.name} — downloading...`);
            try {
              const savedPath = await downloadPreview(song.name, previewUrl);
              console.log(` Saved to: ${savedPath}`);
              metadata.push({ query, found: true, path: savedPath });
            } catch (dlErr) {
              console.error(` Error saving preview for ${query}:`, dlErr.message || dlErr);
              metadata.push({ query, found: false, error: dlErr.message || String(dlErr) });
            }
          } else {
            console.log(` No preview URL returned by preview-finder for: ${query}`);
            metadata.push({ query, found: false, note: 'no preview url from preview-finder' });
          }
        } else {
          console.log(` No results from preview-finder for: ${query}`);
          metadata.push({ query, found: false, note: 'no results from preview-finder' });
        }
      } catch (finderErr) {
        console.error(` spotify-preview-finder failed for "${query}":`, finderErr.message || finderErr);
        metadata.push({ query, found: false, error: finderErr.message || String(finderErr) });
      }
    }

    nextUrl = pageData.next;
  }

  const metaFile = path.join(OUTPUT_DIR, `playlist_${playlistId}_meta.json`);
  fs.writeFileSync(metaFile, JSON.stringify(metadata, null, 2), 'utf8');
  console.log(' Metadata saved to:', metaFile);
}


const argPlaylist = process.argv[2];

async function main() {
  let playlistId = argPlaylist;
  if (!playlistId) {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });

    playlistId = await new Promise((resolve) =>
      rl.question(' Enter Spotify Playlist ID: ', (answer) => {
        rl.close();
        resolve(answer.trim());
      })
    );
  }

  if (!playlistId) {
    console.error('❌ No playlist ID provided.');
    process.exit(1);
  }


  playlistId = playlistId.replace(/^.*playlist\//, "").split("?")[0];

  await fetchPlaylistAndDownloadPreviews(playlistId);
  console.log(' Done.');
}

main().catch(err => {
  console.error(' Fatal error:', err && err.message ? err.message : err);
  process.exit(1);
});
