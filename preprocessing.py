import urllib
import urllib.request
from data_types import Song
import re
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
from scipy.io import wavfile
import noisereduce as nr

# Thread-safe counter for progress tracking
_lock = threading.Lock()
_done = 0

def scrape_site(url, max_workers):
    global _done
    _done = 0

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode('utf-8', errors='ignore')

    # ── Phase 1: Parse all metadata from the index page (fast, no I/O) ──
    entries = []
    blocks = re.split(r'(?i)<li[^>]*>', html)[1:]
    total = len(blocks)

    for block in blocks:
        title_block = re.search(r'<b>(.*?)</b>', block, re.S | re.I)
        if not title_block:
            continue

        clean_text = re.sub(r'<[^>]+>', '', title_block.group(1)).strip()
        clean_text = re.sub(r'\s+', ' ', clean_text)

        colon_idx = clean_text.find(':')
        if colon_idx == -1:
            continue

        title = clean_text[:colon_idx].strip()
        remainder = clean_text[colon_idx + 1:].strip()

        parts = [p.strip() for p in remainder.split(';')]
        if len(parts) < 3:
            continue

        ragam = parts[0]
        talam = parts[1]
        composer = parts[2]

        not_match = re.search(r'href="([^"]*)"[^>]*>\s*Notation', block, re.I)
        aud_match = re.search(
            r'href="([^"]*(?:class|detailed)[^"]*\.mp3)"', block, re.I
        )
        if not aud_match:
            aud_match = re.search(r'href="([^"]*\.mp3)"', block, re.I)

        n_url = urljoin(url, not_match.group(1)) if not_match else None
        a_url = urljoin(url, aud_match.group(1)) if aud_match else None

        entries.append((title, composer, ragam, talam, a_url, n_url))

    song_count = len(entries)
    print(f"Parsed {song_count}/{total} entries. Fetching notation pages ({max_workers} threads)...\n")

    # ── Phase 2: Fetch all notation pages concurrently ──
    def _fetch_one(idx, entry):
        global _done
        title, composer, ragam, talam, a_url, n_url = entry
        lyr, mng = scrape_notation(n_url) if n_url else ("", "")

        with _lock:
            _done += 1
            status = []
            if a_url: status.append("audio")
            if n_url: status.append("notation")
            if lyr:   status.append("lyrics")
            if mng:   status.append("meaning")
            print(f"  [{_done}/{song_count}] {title} — {', '.join(status) if status else 'metadata only'}")

        return idx, Song(title, composer, ragam, talam, None, "Krithi", "Shivkumar", a_url, lyr, mng)

    songs = [None] * song_count
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, i, e): i for i, e in enumerate(entries)}
        for future in as_completed(futures):
            idx, song = future.result()
            songs[idx] = song

    print(f"\nDone! {song_count} songs scraped.")
    print(f"  Audio: {sum(1 for s in songs if s.audio_url)}, "
          f"Lyrics: {sum(1 for s in songs if s.lyrics)}, "
          f"Meanings: {sum(1 for s in songs if s.meaning)}")
    return songs

def scrape_notation(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
    except Exception:
        return "", ""

    def clean(t): return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', t)).strip()

    # Lyrics: extract text between the first "Paasuram" <h1> heading and
    # the next section marker (Meaning, Context, or Sahityam)
    lyr = re.search(
        r'Paasuram.*?</h1>(.*?)(?:<[^>]*>(?:Meaning|Context|Sahityam))',
        html, re.S | re.I
    )

    # Meaning/Context: find the first Context or Meaning header with </u></i>
    # and extract until the next <h1> or end of document
    mng = re.search(
        r'(?:Meaning|Context).*?</u></i>(.*?)(?:<h1|<hr|\Z)',
        html, re.S | re.I
    )

    return (clean(lyr.group(1)) if lyr else ""), (clean(mng.group(1)) if mng else "")

def convert_audio_links(songs, max_workers):
    import os
    os.makedirs("songs", exist_ok=True)

    downloadable = [(i, s) for i, s in enumerate(songs) if s.audio_url]
    total = len(downloadable)
    done = 0
    done_lock = threading.Lock()

    print(f"\nDownloading {total} audio files ({max_workers} threads)...\n")

    def _download(idx, song):
        nonlocal done
        filename = "songs/" + song.title + "_" + song.composer + ".mp3"
        try:
            urllib.request.urlretrieve(song.audio_url, filename)
            song.audio_file = filename
            with done_lock:
                done += 1
                print(f"  [{done}/{total}] Downloaded: {song.title}")
        except Exception:
            with done_lock:
                done += 1
                print(f"  [{done}/{total}] FAILED: {song.title}")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_download, i, s) for i, s in downloadable]
        for f in as_completed(futures):
            f.result()  # propagate any uncaught exceptions

    success = sum(1 for s in songs if s.audio_file)
    print(f"\nDownloads complete: {success}/{total} succeeded.")

def clear_incomplete_songs(songs):
    complete = []
    removed_reasons = {"no_file": 0, "bad_ext": 0, "missing": 0, "too_small": 0}
    for s in songs:
        if not s.audio_file:
            removed_reasons["no_file"] += 1
            continue
        if not (s.audio_file.endswith(".mp3") or s.audio_file.endswith(".wav")):
            removed_reasons["bad_ext"] += 1
            continue
        if not os.path.exists(s.audio_file):
            removed_reasons["missing"] += 1
            continue
        if os.path.getsize(s.audio_file) < 1024:
            removed_reasons["too_small"] += 1
            continue
        complete.append(s)
    removed = len(songs) - len(complete)
    print(f"Removed {removed} incomplete songs ({len(complete)} remaining).")
    for reason, count in removed_reasons.items():
        if count > 0:
            print(f"  - {reason}: {count}")
    return complete

def minimize_static(songs, max_workers):
    wav_songs = [s for s in songs if s.audio_file and s.audio_file.endswith(".wav")]
    total = len(wav_songs)
    done = 0
    done_lock = threading.Lock()

    if total == 0:
        print("\nNo .wav files to process for static minimization.")
        return

    print(f"\nMinimizing static for {total} audio files ({max_workers} threads)...\n")

    def _minimize(idx, song):
        nonlocal done
        try:
            import numpy as np
            rate, data = wavfile.read(song.audio_file)
            original_dtype = data.dtype

            # noisereduce expects shape (channels, samples) for multi-channel
            # but scipy.wavfile.read returns (samples, channels)
            if data.ndim == 2:
                data = data.T  # transpose to (channels, samples)
                reduced_noise = nr.reduce_noise(y=data.astype(np.float64), sr=rate)
                reduced_noise = reduced_noise.T  # transpose back to (samples, channels)
            else:
                reduced_noise = nr.reduce_noise(y=data.astype(np.float64), sr=rate)

            # Preserve original dtype (e.g. int16) for WAV compatibility
            reduced_noise = np.clip(reduced_noise, 
                                    np.iinfo(original_dtype).min, 
                                    np.iinfo(original_dtype).max).astype(original_dtype)

            wavfile.write(song.audio_file, rate, reduced_noise)
            with done_lock:
                done += 1
                print(f"  [{done}/{total}] Minimized static: {song.title}")
        except Exception as e:
            with done_lock:
                done += 1
                print(f"  [{done}/{total}] FAILED static minimization: {song.title} - {e}")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_minimize, i, s) for i, s in enumerate(wav_songs)]
        for f in as_completed(futures):
            f.result()  # propagate any uncaught exceptions

    print(f"\nStatic minimization complete.")
        
def convert_audio_wav(songs, max_workers):
    import os
    os.makedirs("songs", exist_ok=True)

    convertible = [(i, s) for i, s in enumerate(songs) if s.audio_file and s.audio_file.endswith(".mp3")]
    total = len(convertible)
    done = 0
    done_lock = threading.Lock()

    print(f"\nConverting {total} audio files ({max_workers} threads)...\n")

    def _convert(idx, song):
        nonlocal done
        filename = "songs/" + song.title + "_" + song.composer + ".wav"
        try:
            import subprocess
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run([ffmpeg_exe, '-y', '-i', song.audio_file, filename], 
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            song.audio_file = filename
            with done_lock:
                done += 1
                print(f"  [{done}/{total}] Converted: {song.title}")
        except Exception as e:
            with done_lock:
                done += 1
                print(f"  [{done}/{total}] FAILED: {song.title} - Error: {e}")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_convert, i, s) for i, s in convertible]
        for f in as_completed(futures):
            f.result()  # propagate any uncaught exceptions

    success = sum(1 for s in songs if s.audio_file and s.audio_file.endswith(".wav"))
    print(f"\nConversion complete: {success}/{total} succeeded.")



    