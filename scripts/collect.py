# =============================================================
# CWA-MIR Corpus Builder  ·  collect.py
# Caribbean & West African Music Information Retrieval Study
# =============================================================
# Data sources:
#   Spotify Web API  — metadata, catalog search, artist info
#   iTunes Search API — 30s audio preview download (no auth needed)
#
# Auth required (Spotify only):
#   SPOTIFY_CLIENT_ID     → https://developer.spotify.com/dashboard
#   SPOTIFY_CLIENT_SECRET → same dashboard → Settings → Client Secret
#   (Client Credentials flow — no user login required)
#
# iTunes API:
#   No authentication. No API key. Rate limit: 20 req/min.
#   Returns 30-second AAC (.m4a) preview clips for most catalog tracks.
#   Docs: https://developer.apple.com/library/archive/documentation/
#         AudioVideo/Conceptual/iTuneSearchAPI/
#
# Run:
#   source .venv/bin/activate
#   python scripts/collect.py
#
# Resumable: re-running skips already-collected track IDs.
# =============================================================

import os
import json
import time
import re
import unicodedata
import difflib
import requests
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from tqdm import tqdm
import spotipy
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials

# %% [Cell 1] ─── Path Setup & Environment ───────────────────
# =============================================================
# Phase 1 ── Path Setup & Environment
# =============================================================

load_dotenv()

ROOT  = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "data" / "raw" / "audio"
META  = ROOT / "data" / "metadata" / "corpus.jsonl"

AUDIO.mkdir(parents=True, exist_ok=True)
META.parent.mkdir(parents=True, exist_ok=True)

_cid = os.getenv("SPOTIFY_CLIENT_ID")
_sec = os.getenv("SPOTIFY_CLIENT_SECRET")

if not _cid or not _sec:
    print("ERROR: Spotify credentials not found in .env")
    print("   SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET are required.")
    print("   Get them at: https://developer.spotify.com/dashboard")
    exit(1)

print(f"Environment loaded (Client ID: {_cid[:6]}...)")


# %% [Cell 2] ─── API Clients & Connectivity Checks ──────────
# =============================================================
# Phase 2 ── API Clients
# =============================================================

# ── Spotify client (Client Credentials flow) ─────────────────
# Authenticates as the app, not a user. Provides full read access
# to the public catalog: search, track metadata, artist info,
# preview URLs. No Premium account required.
sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=_cid,
        client_secret=_sec,
    ),
    requests_timeout=10,
    retries=3,
)

try:
    _test = sp.search(q="reggae", type="track", limit=1)
except SpotifyException as e:
    print(f"Spotify search is unavailable: {e}")
    exit(1)
if not _test["tracks"]["items"]:
    print("Spotify authentication returned empty. Check API permissions.")
    exit(1)
print("Spotify authentication confirmed")

# ── iTunes API (no auth) ──────────────────────────────────────
# The iTunes Search API is a public REST endpoint.
# No API key. No OAuth. Rate limit: 20 requests per minute.
# We sleep 3.1 seconds between iTunes calls to stay safely under.
ITUNES_BASE    = "https://itunes.apple.com"
ITUNES_HEADERS = {
    "User-Agent": "CWA-MIR-Research/1.0 (corpus-builder; academic use)",
    "Accept":     "application/json",
}
ITUNES_RATE_SLEEP = 3.1   # seconds between iTunes requests (20/min limit)

_itunes_test = requests.get(
    f"{ITUNES_BASE}/search",
    params={"term": "reggae", "media": "music", "limit": 1, "country": "US"},
    headers=ITUNES_HEADERS,
    timeout=10,
)
if _itunes_test.status_code == 200:
    print("iTunes API connection confirmed")
else:
    print(f"WARNING: iTunes API returned {_itunes_test.status_code} — preview download may fail")
    print("   Collection will continue; tracks without previews are logged.")


# %% [Cell 3] ─── Data Structures ────────────────────────────
# =============================================================
# Phase 3 ── Data Structures
# =============================================================

@dataclass
class Track:
    """
    One row in corpus.jsonl. Every field maps to a metadata column.

    Metadata comes from Spotify (authoritative catalog source).
    Audio preview comes from iTunes (30s AAC clip, no auth needed).
    The two are linked by artist+title fuzzy match — see itunes_lookup().
    """
    # ── Identity ──────────────────────────────────────────────
    track_id:        str          # Spotify track ID — stable, unique, catalog-linkable
    title:           str
    artist:          str
    artist_id:       str          # Spotify artist ID (for related-artist expansion)

    # ── Classification ────────────────────────────────────────
    genre:           str          # genre label used to collect this track
    country_origin:  str          # ISO 3166-1 alpha-2 or "UNK"
    decade:          str          # e.g. "1980s" | "unknown"
    corpus:          str          # "caribbean" | "westafrican"
    tradition:       str  = ""    # West African corpus only

    # ── Spotify metadata ──────────────────────────────────────
    spotify_url:     str  = ""
    spotify_markets: str  = ""    # pipe-separated ISO market codes
    duration_ms:     int  = 0     # full track duration
    popularity:      int  = 0     # Spotify popularity score 0–100
    release_date:    str  = ""    # as returned by Spotify (YYYY or YYYY-MM-DD)

    # ── iTunes audio ──────────────────────────────────────────
    itunes_id:       int  = 0     # iTunes trackId (0 if no match found)
    itunes_url:      str  = ""    # iTunes store page URL
    itunes_genre:    str  = ""    # iTunes primaryGenreName (cross-reference signal)
    preview_path:    str  = ""    # relative path to downloaded .m4a file
    preview_source:  str  = ""    # "itunes" | "spotify" | "" (none available)

    # ── MusicBrainz metadata ──────────────────────────────────
    isrc:            str  = ""    # ISRC retrieved from Spotify
    mb_genres:       str  = ""    # pipe-separated MusicBrainz genres/tags
    mb_country:      str  = ""    # release country code from MusicBrainz
    spotify_artist_genres: str = "" # pipe-separated artist genres from Spotify

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# %% [Cell 4] ─── Helpers ────────────────────────────────────
# =============================================================
# Phase 4 ── Helpers
# =============================================================

# ── General utilities ─────────────────────────────────────────

def decade_from(release_date: str) -> str:
    """
    Convert a Spotify release_date string to a decade label.
    Spotify returns 'YYYY', 'YYYY-MM', or 'YYYY-MM-DD'.
    Returns 'unknown' on any invalid/empty input.
    """
    if not release_date:
        return "unknown"
    year_str = release_date[:4]
    if not year_str.isdigit() or len(year_str) < 4:
        return "unknown"
    return year_str[:3] + "0s"


def load_seen_ids() -> set:
    """
    Return all track_ids already in corpus.jsonl.
    Handles blank lines and malformed entries — a crash mid-write
    never prevents a clean resume on the next run.
    """
    if not META.exists():
        return set()
    seen = set()
    with META.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["track_id"])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  WARNING: corpus.jsonl line {lineno} skipped ({e})")
    return seen


def append_track(t: Track) -> None:
    """Append one track to corpus.jsonl (atomic single-line write)."""
    with META.open("a", encoding="utf-8") as f:
        f.write(t.to_jsonl() + "\n")


def count_for(field_name: str, value: str) -> int:
    """Count tracks in corpus.jsonl where field_name == value."""
    if not META.exists():
        return 0
    n = 0
    with META.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                if json.loads(line).get(field_name) == value:
                    n += 1
            except json.JSONDecodeError:
                pass
    return n


# ── Spotify helpers ───────────────────────────────────────────

def safe_search(query: str, market: str = "US", limit: int = 10) -> list:
    """
    Wrapper around sp.search with retry + rate-limit handling.
    Returns the list of track items or [] on all failures.
    Spotify Client Credentials allows ~180 requests/minute.
    """
    for attempt in range(3):
        try:
            results = sp.search(
                q=query, type="track", market=market, limit=limit
            )
            return results["tracks"]["items"]
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:
                wait = int(getattr(e, "headers", {}).get("Retry-After", 5))
                print(f"  WARNING: Spotify rate limit — waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"  FAILED: Spotify error (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  FAILED: Unexpected error (attempt {attempt + 1}): {e}")
            time.sleep(2 ** attempt)
    return []


# Global artist genre cache to minimize redundant Spotify API calls
ARTIST_GENRE_CACHE = {}

def get_spotify_artist_genres(artist_id: str) -> list:
    """
    Retrieve and cache artist genres from Spotify.
    Keeps API calls minimal by checking the global cache first.
    """
    if not artist_id:
        return []
    if artist_id in ARTIST_GENRE_CACHE:
        return ARTIST_GENRE_CACHE[artist_id]

    for attempt in range(3):
        try:
            artist_info = sp.artist(artist_id)
            genres = artist_info.get("genres", [])
            ARTIST_GENRE_CACHE[artist_id] = genres
            return genres
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:
                wait = int(getattr(e, "headers", {}).get("Retry-After", 5))
                print(f"  WARNING: Spotify rate limit — waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"  FAILED: Spotify error fetching artist (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  FAILED: Unexpected error fetching artist (attempt {attempt + 1}): {e}")
            time.sleep(2 ** attempt)
            
    ARTIST_GENRE_CACHE[artist_id] = []
    return []


# ── iTunes helpers ────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Standardize text for high-accuracy string matching:
    1. Convert to lowercase.
    2. Normalize unicode characters (remove accents/diacritics).
    3. Standardize common abbreviations and punctuation (e.g., '&' -> 'and').
    4. Strip out common descriptive suffixes (e.g. '(Remastered)', '[Radio Edit]').
    5. Remove non-alphanumeric characters.
    6. Collapse multiple spaces into one.
    """
    if not text:
        return ""
    # Lowercase & strip
    text = text.lower().strip()
    # Deconstruct accented characters (e.g. "Adé" -> "Ade")
    text = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    # Convert & to and
    text = text.replace("&", "and")
    # Clean up standard song variations/edits from titles
    text = re.sub(r'\(remastered.*?\)|\[remastered.*?\]|\(radio edit\)|- remastered.*', '', text)
    text = re.sub(r'\(single version\)|- single version|\(album version\)|- album version', '', text)
    text = re.sub(r'\(feat\..*?\)|\[feat\..*?\]|\(featuring.*?\)|\[featuring.*?\]', '', text)
    # Remove all punctuation and symbols, keeping letters, numbers, and single spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Normalize spacing
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _match_score(spotify_artist: str, spotify_title: str,
                 itunes_artist: str, itunes_title: str) -> float:
    """
    Fuzzy similarity score between a Spotify track and an iTunes result.

    Weights: artist 60%, title 40%.
    Artist is weighted higher to prevent wrong-artist matches when a
    title is common (e.g. 'Hot Hot Hot' exists for multiple artists).

    Handles Unicode artist name variants (e.g. 'King Sunny Adé' vs
    'King Sunny Ade') and punctuation differences ('Jean & Dinah'
    vs 'Jean and Dinah') naturally via normalize_text().

    Threshold: 0.75 — empirically validated across Caribbean and
    West African artist names. Accepts:
      'Arrow / Hot Hot Hot'  vs  'Arrow / Hot Hot Hot (Remastered)'  → 1.00
      'King Sunny Ade / Synchro System'  vs  'King Sunny Adé / ...'  → 1.00
    Rejects:
      'Arrow / Hot Hot Hot'  vs  'Bow Wow / Hot Hot Hot'             → ~0.40
    """
    s_artist = normalize_text(spotify_artist)
    s_title = normalize_text(spotify_title)
    i_artist = normalize_text(itunes_artist)
    i_title = normalize_text(itunes_title)

    # Use difflib SequenceMatcher on normalized strings
    artist_score = difflib.SequenceMatcher(None, s_artist, i_artist).ratio()
    title_score = difflib.SequenceMatcher(None, s_title, i_title).ratio()

    # Handle cases where one artist contains the other (e.g., featuring artists listed in iTunes)
    if s_artist in i_artist or i_artist in s_artist:
        artist_score = max(artist_score, 0.95)

    return (artist_score * 0.6) + (title_score * 0.4)


def itunes_lookup(artist: str, title: str) -> dict:
    """
    Search iTunes for a track matching the given artist and title.
    Returns the best matching iTunes result dict, or {} on no match.

    Flow:
    1. GET /search?term={artist}+{title}&media=music&entity=song&limit=5
    2. Score each result with _match_score()
    3. Accept the highest-scoring result if score >= 0.75
    4. Sleep ITUNES_RATE_SLEEP seconds after every call (20 req/min limit)

    The iTunes previewUrl field in the result is a 30-second AAC clip
    served directly from Apple's CDN — no additional auth required.
    """
    term = f"{artist} {title}"
    try:
        r = requests.get(
            f"{ITUNES_BASE}/search",
            params={
                "term":    term,
                "media":   "music",
                "entity":  "song",
                "limit":   5,
                "country": "US",
            },
            headers=ITUNES_HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
    except Exception as e:
        print(f"  FAILED: iTunes search failed for '{term}': {e}")
        results = []
    finally:
        time.sleep(ITUNES_RATE_SLEEP)   # always sleep, even on error

    if not results:
        return {}

    # Score all results, return best match above threshold
    best, best_score = {}, 0.0
    for result in results:
        if result.get("kind") != "song":
            continue
        score = _match_score(
            artist, title,
            result.get("artistName", ""),
            result.get("trackName", ""),
        )
        if score > best_score:
            best_score = score
            best = result

    return best if best_score >= 0.75 else {}


# ── MusicBrainz helpers ───────────────────────────────────────

MUSICBRAINZ_BASE = "https://musicbrainz.org/ws/2"
MUSICBRAINZ_HEADERS = {
    "User-Agent": "CWA-MIR-Research/1.0 (corpus-builder; mailto:contact-email@example.com)"
}
MUSICBRAINZ_RATE_SLEEP = 1.1

def musicbrainz_lookup(isrc: str, artist: str, title: str) -> dict:
    """
    Query MusicBrainz for track genre and country of origin.
    First tries an ISRC lookup (highly accurate), then falls back to artist+title search.
    Enforces strict 1-second rate limit to prevent 503 errors.
    """
    mb_genres = []
    mb_country = ""

    # Helper to extract tags and countries from a list of recording dicts
    def extract_from_recordings(recordings):
        nonlocal mb_country
        genres_set = set()
        for rec in recordings:
            # Extract genres
            for g in rec.get("genres", []):
                genres_set.add(g.get("name", "").lower())
            # Extract tags (sometimes genres are stored as tags)
            for t in rec.get("tags", []):
                genres_set.add(t.get("name", "").lower())
            
            # Extract release country (releases have 2-letter country codes)
            if not mb_country:
                for release in rec.get("releases", []):
                    c = release.get("country", "")
                    if c and len(c) == 2 and c != "XW": # XW means 'World'
                        mb_country = c
                        break
        return list(genres_set)

    # 1. Try ISRC lookup first
    if isrc:
        try:
            r = requests.get(
                f"{MUSICBRAINZ_BASE}/isrc/{isrc}",
                params={"inc": "genres+tags+releases", "fmt": "json"},
                headers=MUSICBRAINZ_HEADERS,
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            recordings = data.get("recordings", [])
            if recordings:
                mb_genres = extract_from_recordings(recordings)
        except Exception as e:
            # Quietly fail and let fallback search attempt
            pass
        finally:
            time.sleep(MUSICBRAINZ_RATE_SLEEP)

    # 2. Fallback to artist + title search if ISRC yielded nothing
    if not mb_genres:
        # Clean artist/title terms for search query
        clean_artist = re.sub(r'[^a-zA-Z0-9\s]', '', artist).strip()
        clean_title = re.sub(r'[^a-zA-Z0-9\s]', '', title).strip()
        query = f'artist:"{clean_artist}" AND recording:"{clean_title}"'
        try:
            r = requests.get(
                f"{MUSICBRAINZ_BASE}/recording",
                params={"query": query, "fmt": "json"},
                headers=MUSICBRAINZ_HEADERS,
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            recordings = data.get("recordings", [])
            if recordings:
                mb_genres = extract_from_recordings(recordings)
        except Exception as e:
            # Quietly fail
            pass
        finally:
            time.sleep(MUSICBRAINZ_RATE_SLEEP)

    return {
        "genres": [g for g in mb_genres if g],
        "country": mb_country
    }


def download_preview(url: str, path: Path, label: str = "") -> bool:
    """
    Download an iTunes preview AAC clip or Spotify MP3 to disk.
    iTunes previewUrls and Spotify previews are direct CDN links — no auth header needed.
    Rejects responses under 1 KB to catch empty-body 200s.
    """
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        if len(r.content) < 1024:
            print(f"  WARNING: Preview too small ({len(r.content)}B) — skipped {label}")
            return False
        path.write_bytes(r.content)
        return True
    except requests.RequestException as e:
        print(f"  FAILED: Download failed {label}: {e}")
        return False


# ── Core processing loop ──────────────────────────────────────

def process_spotify_items(
    items:         list,
    genre:         str,
    corpus:        str,
    tradition:     str,
    market_filter: list,  # empty = accept all; list of ISO codes = require intersection
    seen:          set,
    pbar,
    target:        int,
) -> None:
    """
    Shared inner loop used by both Caribbean and West African passes.

    For each Spotify track item:
    1. Skip if already collected or market filter not satisfied
    2. Look up the track on iTunes by artist + title
    3. Download the iTunes 30s preview (.m4a) if found
    4. Fall back to Spotify's own preview_url if iTunes has no match
       (Spotify previews are also 30s MP3 clips — usable but lower
       priority since they may expire; iTunes CDN links are stable)
    5. Write one record to corpus.jsonl

    market_filter: list of Caribbean ISO market codes. A Spotify track
    passes if ANY of its available_markets is in the list. This is the
    most reliable automated proxy for Caribbean origin without a manual
    artist country lookup.
    """
    for item in items:
        if pbar.n >= target:
            break

        track_id = item["id"]
        if track_id in seen:
            continue

        # Market filter (Caribbean pass only)
        # NOTE: Spotify deprecated and removed the available_markets field in Feb 2026 
        # for unextended credentials, so it always returns empty. We bypass this filter 
        # and rely on the query's targeted search market to ensure regional relevance.
        track_markets = set(item.get("available_markets", []))
        # if market_filter and not track_markets.intersection(market_filter):
        #     continue

        artist       = item["artists"][0]["name"]
        artist_id    = item["artists"][0]["id"]
        title        = item["name"]
        release_date = item["album"].get("release_date", "")
        spotify_preview = item.get("preview_url", "")
        isrc         = item.get("external_ids", {}).get("isrc", "")

        # ── Spotify Artist lookup ─────────────────────────────
        spotify_artist_genres = get_spotify_artist_genres(artist_id)

        # ── MusicBrainz lookup ────────────────────────────────
        mb_data = musicbrainz_lookup(isrc, artist, title)
        mb_genres = mb_data["genres"]
        mb_country = mb_data["country"]

        # ── iTunes lookup ─────────────────────────────────────
        itunes_data    = itunes_lookup(artist, title)
        itunes_preview = itunes_data.get("previewUrl", "")
        itunes_id      = int(itunes_data.get("trackId", 0))
        itunes_url     = itunes_data.get("trackViewUrl", "")
        itunes_genre   = itunes_data.get("primaryGenreName", "")

        # Create our Consensus Pool of tags (lowercase and stripped)
        raw_tags = [g.lower().strip() for g in mb_genres] + [g.lower().strip() for g in spotify_artist_genres]
        if itunes_genre:
            raw_tags.append(itunes_genre.lower().strip())
        
        # Keep full tag phrases plus split words for broad coverage
        consensus_pool = set()
        for tag in raw_tags:
            if not tag:
                continue
            consensus_pool.add(tag)
            # Add split words to catch nested genres (e.g. 'bouyon soca' -> 'bouyon', 'soca')
            for word in re.split(r'[^a-z0-9]', tag):
                if word:
                    consensus_pool.add(word)
        
        # Validation Logic:
        # Rule 1: Positive Match Override
        # If the target genre is explicitly in the consensus pool, we ACCEPT immediately
        has_target_tag = any(genre.lower() in tag or tag in genre.lower() for tag in consensus_pool)

        if consensus_pool and not has_target_tag:
            # Rule 2: Explicit Mismatch Rejection
            # If target genre is not in the pool, check if any of its specific conflicting genres is in the pool
            conflicts = GENRE_CONFLICTS.get(genre.lower(), set())
            has_other_tag = any(any(conflict in tag for conflict in conflicts) for tag in consensus_pool)
            if has_other_tag:
                tqdm.write(f"  WARNING: Skipping explicit mismatch (Consensus Pool): '{artist} – {title}' has tags {consensus_pool} (target: {genre})")
                continue

        preview_path   = ""
        preview_source = ""

        # Ensure genre subfolder exists
        genre_dir = AUDIO / genre
        genre_dir.mkdir(parents=True, exist_ok=True)

        if itunes_preview:
            # Prefer iTunes: stable CDN URL, AAC quality, no expiry
            audio_path = genre_dir / f"{track_id}.m4a"
            if not audio_path.exists():
                if download_preview(itunes_preview, audio_path, f"{artist} – {title}"):
                    preview_path   = str(audio_path.relative_to(ROOT))
                    preview_source = "itunes"
            else:
                preview_path   = str(audio_path.relative_to(ROOT))
                preview_source = "itunes"

        elif spotify_preview:
            # Fallback: Spotify's own preview_url (30s MP3, may expire)
            audio_path = genre_dir / f"{track_id}.mp3"
            if not audio_path.exists():
                if download_preview(spotify_preview, audio_path, f"{artist} – {title}"):
                    preview_path   = str(audio_path.relative_to(ROOT))
                    preview_source = "spotify"
            else:
                preview_path   = str(audio_path.relative_to(ROOT))
                preview_source = "spotify"

        # Skip if no audio could be obtained — corpus requires paired audio
        if not preview_path:
            continue

        # Use MusicBrainz country of origin if found, fallback to UNK
        country_origin = mb_country if mb_country else "UNK"

        track = Track(
            track_id        = track_id,
            title           = title,
            artist          = artist,
            artist_id       = artist_id,
            genre           = genre,
            country_origin  = country_origin,
            decade          = decade_from(release_date),
            corpus          = corpus,
            tradition       = tradition,
            spotify_url     = item["external_urls"].get("spotify", ""),
            spotify_markets = "|".join(sorted(track_markets)),
            duration_ms     = item.get("duration_ms", 0),
            popularity      = item.get("popularity", 0),
            release_date    = release_date,
            itunes_id       = itunes_id,
            itunes_url      = itunes_url,
            itunes_genre    = itunes_genre,
            preview_path    = preview_path,
            preview_source  = preview_source,
            isrc            = isrc,
            mb_genres       = "|".join(mb_genres),
            mb_country      = mb_country,
            spotify_artist_genres = "|".join(spotify_artist_genres),
        )

        append_track(track)
        seen.add(track_id)
        pbar.update(1)
        time.sleep(0.1)


# %% [Cell 5] ─── Corpus Target Definitions ──────────────────
# =============================================================
# Phase 5 ── Corpus Target Definitions
# =============================================================

# Map of target genres to their disallowed explicit conflicting genres.
# This prevents overly broad, coarse bucket classifications (like iTunes labeling tracks as 'Afrobeats' or 'Reggae')
# from causing false negatives, while still strictly blocking genuine mismatches (like Dancehall slipping into Soca).
GENRE_CONFLICTS = {
    # ── Caribbean ─────────────────────────────────────────────
    "compas":            {"reggae_dancehall", "calypso_soca"},
    "zouk":              {"reggae_dancehall", "calypso_soca"},
    "reggae_dancehall":  {"calypso_soca", "compas", "zouk"},
    "calypso_soca":      {"reggae_dancehall", "compas", "zouk"},

    # ── West African ──────────────────────────────────────────
    "juju":      {"afrobeats", "mbalax", "kora"},
    "fuji":      {"afrobeats", "mbalax", "kora"},
    "apala":     {"afrobeats", "mbalax", "kora"},
    "highlife":  {"afrobeats", "mbalax"},
    "kora":      {"afrobeats", "highlife", "mbalax"},
    "mbalax":    {"afrobeats", "highlife", "kora"},
    "afrobeats": set(), # Afrobeats is too broad/modern to conflict
}

# Caribbean market codes — used to filter Spotify results by
# distribution region as a proxy for Caribbean artist origin.
CARIB_MARKETS = {
    "BB", "DM", "GD", "GP", "HT", "JM", "LC", "MQ",
    "TT", "VC", "KN", "AG", "DO", "CU", "PR", "AW", "MS",
}

CARIBBEAN = {
    "genres": ["compas", "zouk", "reggae_dancehall", "calypso_soca"],
    "per_genre": 75,
    # Spotify free-text search (not the undocumented genre: filter).
    # Queries ordered from most specific to broadest fallback.
    "queries": {
        "compas":            ["compas haiti", "haitian compas", "kompa music", "kompa direct"],
        "zouk":              ["zouk martinique", "zouk caribbean", "zouk music"],
        "reggae_dancehall":  ["reggae jamaica", "roots reggae", "reggae music", "dancehall jamaica", "dancehall reggae", "dancehall"],
        "calypso_soca":      ["calypso trinidad", "calypso music caribbean", "calypso", "soca music trinidad", "soca carnival", "soca"],
    },
    "search_markets": ["TT", "JM", "BB", "US"],
}

WEST_AFRICAN = {
    "per_tradition": 30,
    "traditions": [
        {
            "name":    "yoruba",
            "genres":  ["juju", "fuji", "apala"],
            "queries": [
                "juju music nigeria", "king sunny ade juju",
                "fuji music nigeria", "apala music yoruba",
                "ebenezer obey juju",
            ],
            "artist_seeds": [
                "King Sunny Ade", "Ebenezer Obey", "Fela Kuti",
                "Sikiru Ayinde Barrister", "Haruna Ishola", "Sir Shina Peters",
            ],
        },
        {
            "name":    "akan",
            "genres":  ["highlife"],
            # NOTE: Fontomfrom and Atumpan are percussion traditions with no
            # Spotify presence. Excluded from active collection due to API gaps.
            "queries": [
                "ghana highlife", "highlife music ghana", "ebo taylor highlife",
                "african highlife", "pat thomas ghana",
            ],
            "artist_seeds": [
                "Ebo Taylor", "E.T. Mensah", "Pat Thomas",
                "Gyedu-Blay Ambolley", "Amakye Dede", "Alex Konadu",
            ],
        },
        {
            "name":    "mande",
            "genres":  ["kora", "mbalax"],
            # NOTE: Traditional oral Griot song is a stylistic gap on commercial APIs.
            # Collection is focused on acoustic Kora music and Mbalax.
            "queries": [
                "kora music mali", "toumani diabate kora",
                "mbalax senegal", "youssou ndour mbalax",
                "baaba maal senegal",
            ],
            "artist_seeds": [
                "Toumani Diabaté", "Youssou N'Dour", "Baaba Maal",
                "Salif Keita", "Ali Farka Touré", "Habib Koité",
            ],
        },
        {
            "name":    "modern_afrobeats",
            "genres":  ["afrobeats", "afropop"],
            "queries": [
                "afrobeats nigeria", "afropop ghana",
                "burna boy afrobeats", "wizkid afrobeats",
                "davido afrobeats", "afrobeats 2020",
            ],
            "artist_seeds": [
                "Burna Boy", "Wizkid", "Davido",
                "Sarkodie", "Tiwa Savage", "Mr Eazi",
            ],
        },
    ],
}


# %% [Cell 6] ─── Collection Logic ───────────────────────────
# =============================================================
# Phase 6 ── Collection Logic
# =============================================================

def collect_caribbean(seen: set) -> None:
    """
    Caribbean corpus collection.

    Spotify provides the track list and metadata.
    iTunes provides the 30-second audio preview clip.

    For each genre:
    1. Try each query string across multiple Caribbean + US search markets
    2. Filter results by available_markets ∩ CARIB_MARKETS (at least one hit)
    3. For each passing track → iTunes lookup → download .m4a preview
    4. Fall back to Spotify preview_url if iTunes has no match
    5. Log any shortfall for the coverage report
    """
    print("\n" + "=" * 55)
    print("CARIBBEAN CORPUS PASS")
    print("=" * 55)

    for genre in CARIBBEAN["genres"]:
        have   = count_for("genre", genre)
        target = CARIBBEAN["per_genre"]
        need   = target - have

        if need <= 0:
            print(f"{genre}: already have {have}/{target}")
            continue

        print(f"\n─── {genre} (have {have}, need {need}) ───")
        pbar = tqdm(total=need, desc=genre, unit="track")

        for query in CARIBBEAN["queries"][genre]:
            if pbar.n >= need:
                break
            for market in CARIBBEAN["search_markets"]:
                if pbar.n >= need:
                    break
                items = safe_search(query, market=market, limit=10)
                process_spotify_items(
                    items         = items,
                    genre         = genre,
                    corpus        = "caribbean",
                    tradition     = "",
                    market_filter = list(CARIB_MARKETS),
                    seen          = seen,
                    pbar          = pbar,
                    target        = need,
                )
                time.sleep(0.3)

        if pbar.n < need:
            print(
                f"  WARNING: {genre}: collected {pbar.n}/{need} — "
                f"{need - pbar.n} gap logged for limitations report"
            )
        pbar.close()

    print("\nCaribbean pass complete.")


def collect_west_african(seen: set) -> None:
    """
    West African corpus collection.

    Two-pass strategy per tradition:

    Pass 1 — Keyword queries:
      Genre/artist keyword search on Spotify. No market filter
      (WA markets don't overlap with Caribbean ISO codes).

    Pass 2 — Artist seed fallback:
      Direct artist:"Name" search for curated known artists per
      tradition. Critical for thin-tag genres (Apala, Kora, Griot)
      where keyword search returns insufficient results.
      Artist list is carefully curated — skipping the country filter
      here is intentional and safe given the controlled seed list.

    iTunes lookup runs for every accepted track (same as Caribbean).
    """
    print("\n" + "=" * 55)
    print("WEST AFRICAN CORPUS PASS")
    print("=" * 55)

    for trad in WEST_AFRICAN["traditions"]:
        have   = count_for("tradition", trad["name"])
        target = WEST_AFRICAN["per_tradition"]
        need   = target - have

        if need <= 0:
            print(f"{trad['name']}: already have {have}/{target}")
            continue

        print(f"\n─── {trad['name']} (have {have}, need {need}) ───")
        pbar = tqdm(total=need, desc=trad["name"], unit="track")

        # Pass 1: keyword queries
        for query in trad["queries"]:
            if pbar.n >= need:
                break
            items = safe_search(query, market="US", limit=10)
            process_spotify_items(
                items         = items,
                genre         = trad["genres"][0],
                corpus        = "westafrican",
                tradition     = trad["name"],
                market_filter = [],
                seen          = seen,
                pbar          = pbar,
                target        = need,
            )
            time.sleep(0.3)

        # Pass 2: artist seed fallback
        if pbar.n < need:
            print(f"  → Keyword pass: {pbar.n}/{need}. Running artist fallback...")
            for artist_name in trad["artist_seeds"]:
                if pbar.n >= need:
                    break
                items = safe_search(
                    f'artist:"{artist_name}"', market="US", limit=10
                )
                process_spotify_items(
                    items         = items,
                    genre         = trad["genres"][0],
                    corpus        = "westafrican",
                    tradition     = trad["name"],
                    market_filter = [],
                    seen          = seen,
                    pbar          = pbar,
                    target        = need,
                )
                time.sleep(0.3)

        if pbar.n < need:
            print(
                f"  WARNING: {trad['name']}: collected {pbar.n}/{need} — "
                f"{need - pbar.n} gap logged"
            )
        pbar.close()

    print("\nWest African pass complete.")


# %% [Cell 7] ─── Coverage Report ────────────────────────────
# =============================================================
# Phase 7 ── Coverage Report
# =============================================================

def report_coverage() -> None:
    """
    Read corpus.jsonl and print a structured coverage summary.
    Outputs are formatted for use in the analysis chapter limitations section.
    """
    if not META.exists() or META.stat().st_size == 0:
        print("No data collected yet — run collection first.")
        return

    df = pd.read_json(META, lines=True)
    total = len(df)

    carib_target = CARIBBEAN["per_genre"] * len(CARIBBEAN["genres"])
    wa_target    = WEST_AFRICAN["per_tradition"] * len(WEST_AFRICAN["traditions"])
    grand_total  = carib_target + wa_target

    print("\n" + "=" * 55)
    print(f"CORPUS REPORT  —  {total}/{grand_total} tracks")
    print("=" * 55)

    print("\n── Caribbean by genre ──")
    carib = df[df["corpus"] == "caribbean"]
    if not carib.empty:
        g = carib.groupby("genre").size()
        for genre in CARIBBEAN["genres"]:
            count = int(g.get(genre, 0))
            bar   = "█" * count + "░" * max(0, CARIBBEAN["per_genre"] - count)
            print(f"  {genre:<12} {count:>3}/{CARIBBEAN['per_genre']}  {bar[:25]}")

    print("\n── West African by tradition ──")
    wa = df[df["corpus"] == "westafrican"]
    if not wa.empty:
        g = wa.groupby("tradition").size()
        for trad in WEST_AFRICAN["traditions"]:
            count = int(g.get(trad["name"], 0))
            bar   = "█" * count + "░" * max(0, WEST_AFRICAN["per_tradition"] - count)
            print(f"  {trad['name']:<22} {count:>3}/{WEST_AFRICAN['per_tradition']}  {bar[:25]}")

    print("\n── Audio preview sources ──")
    src = df.groupby("preview_source").size()
    print(f"  iTunes (AAC .m4a)  : {int(src.get('itunes', 0))}")
    print(f"  Spotify (MP3)      : {int(src.get('spotify', 0))}")
    print(f"  None (no preview)  : {int(src.get('', 0) if '' in src else df[df['preview_source']==''].shape[0])}")

    print("\n── By decade ──")
    print(
        df.groupby(["corpus", "decade"])
        .size().unstack(fill_value=0).T.to_string()
    )

    print("\n── Popularity distribution (Spotify 0–100) ──")
    print(df.groupby("corpus")["popularity"].describe().round(1).to_string())

    print("\n── Country of origin ──")
    unk_count = df[df["country_origin"] == "UNK"].shape[0]
    print(f"  UNK (manual enrichment required): {unk_count}/{total}")

    print("\n── LIMITATION NOTES ──")
    gaps = []

    for genre in CARIBBEAN["genres"]:
        n = int(df[df["genre"] == genre].shape[0])
        if n < CARIBBEAN["per_genre"]:
            gaps.append(
                f"  • Caribbean / {genre}: {n}/{CARIBBEAN['per_genre']} — "
                f"{CARIBBEAN['per_genre'] - n} not found via Spotify search"
            )

    for trad in WEST_AFRICAN["traditions"]:
        n = int(df[df["tradition"] == trad["name"]].shape[0])
        if n < WEST_AFRICAN["per_tradition"]:
            gaps.append(
                f"  • West African / {trad['name']}: {n}/{WEST_AFRICAN['per_tradition']} — "
                f"{WEST_AFRICAN['per_tradition'] - n} not found"
            )

    no_audio = df[df["preview_path"] == ""].shape[0]
    if no_audio:
        gaps.append(
            f"  • {no_audio} tracks have no audio preview — "
            f"neither iTunes nor Spotify returned a valid preview URL"
        )

    if unk_count:
        gaps.append(
            f"  • {unk_count} tracks have country_origin=UNK — "
            f"Spotify does not expose artist country; manual enrichment required"
        )

    gaps.append(
        "  • Fontomfrom / Atumpan (Akan percussion): no Spotify/iTunes presence; "
        "supplementary collection from EVIA or Global Jukebox required"
    )
    gaps.append(
        "  • iTunes match uses fuzzy string similarity (threshold 0.75). "
        "A small number of audio files may be alternate versions of the target track"
    )

    for g in gaps:
        print(g)

    print("=" * 55)


def write_genre_specific_metadata() -> None:
    """
    Post-collection hook: Reads the unified corpus.jsonl, groups all successfully 
    collected tracks by genre, and writes out a beautifully formatted, indented 
    genre-specific .json corpus inside each genre's raw folder.
    """
    if not META.exists() or META.stat().st_size == 0:
        return

    print("\nGenerating self-contained genre corpus metadata...")
    genre_tracks = {}
    with META.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                track = json.loads(line)
                g = track.get("genre")
                if g:
                    genre_tracks.setdefault(g, []).append(track)
            except Exception:
                pass

    for g, tracks in genre_tracks.items():
        genre_meta_path = AUDIO / g / f"{g}_corpus.json"
        genre_meta_path.parent.mkdir(parents=True, exist_ok=True)
        with genre_meta_path.open("w", encoding="utf-8") as f:
            json.dump(tracks, f, ensure_ascii=False, indent=2)
        print(f"  Saved self-contained {g}_corpus.json ({len(tracks)} tracks) under raw/{g}/")


# %% [Cell 8] ─── Entry Point ────────────────────────────────
# =============================================================
# Phase 8 ── Entry Point
# =============================================================

if __name__ == "__main__":
    print("\n─── CWA-MIR Collection Pipeline ───")
    print("Metadata : Spotify Web API")
    print("Audio    : iTunes Search API → Spotify preview fallback")

    seen = load_seen_ids()
    print(f"Resuming — {len(seen)} tracks already in corpus\n")

    collect_caribbean(seen)
    collect_west_african(seen)
    report_coverage()
    write_genre_specific_metadata()

    print("\nPipeline complete.")
