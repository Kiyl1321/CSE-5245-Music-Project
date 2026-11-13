import os
import numpy as np
import torch
from torchvggish import vggish, vggish_input
import json

def embed_songs_vggish(songs):
    
    """
    Generate 128-dim VGGish embeddings for all songs with .wav audio files.
    
    For each song:
      1. Convert .wav → log-mel spectrogram frames (~1 sec each)
      2. Run through pre-trained VGGish → (N, 128) tensor
      3. Average across frames → single 128-dim vector
      4. Store on song.embedding and save all to embeddings/song_embeddings.npz
    """
    wav_songs = [s for s in songs if s.audio_file and s.audio_file.endswith(".wav")]
    total = len(wav_songs)

    if total == 0:
        print("\nNo .wav files to embed.")
        return

    print(f"\n{'='*60}")
    print(f"VGGish Embedding: {total} songs")
    print(f"{'='*60}\n")

    # ── Set Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ── Load model once ──
    print("Loading VGGish model...")
    model = vggish()
    model.eval()
    model.to(device)
    
    # Fix torchvggish internal PCA matrices not moving to device automatically
    if hasattr(model, 'pproc'):
        model.pproc._pca_matrix = model.pproc._pca_matrix.to(device)
        model.pproc._pca_means = model.pproc._pca_means.to(device)
        
    print("Model loaded.\n")

    done = 0
    failed = 0

    for song in wav_songs:
        try:
            # Prepare audio
            examples = vggish_input.wavfile_to_examples(song.audio_file)
            
            # Move examples to GPU if available
            if not isinstance(examples, torch.Tensor):
                # Ensure it's a tensor before moving
                examples = torch.from_numpy(examples).float()
            examples = examples.to(device)
            
            # Run model within no_grad to prevent CUDA Out-Of-Memory errors
            with torch.no_grad():
                embeddings = model.forward(examples)  # shape: (N, 128)

            # Average across all ~1-second windows to get a single vector
            if isinstance(embeddings, torch.Tensor):
                embedding_np = embeddings.detach().cpu().numpy()
            else:
                embedding_np = np.array(embeddings)

            song.embedding = np.mean(embedding_np, axis=0)  # shape: (128,)
            done += 1
            print(f"  [{done + failed}/{total}] SUCCESS: {song.title} - embedding shape: {song.embedding.shape}")

        except Exception as e:
            failed += 1
            print(f"  [{done + failed}/{total}] FAILED: {song.title} - {e}")

    # ── Save embeddings to disk ──
    os.makedirs("embeddings", exist_ok=True)
    embeddings_dict = {}
    for song in wav_songs:
        if song.embedding is not None:
            key = f"{song.title}_{song.composer}"
            embeddings_dict[key] = song.embedding

    save_path = os.path.join("embeddings", "song_embeddings.npz")
    np.savez(save_path, **embeddings_dict)

    print(f"\n{'='*60}")
    print(f"Embedding complete: {done}/{total} succeeded, {failed} failed.")
    print(f"Saved {len(embeddings_dict)} embeddings -> {save_path}")
    print(f"Each embedding: 128-dim float vector")
    print(f"{'='*60}\n")

def save_data(songs):
    metadata = []
    for s in songs:
        metadata.append({
            "title": s.title,
            "composer": s.composer,
            "cat": s.cat,
            "ragam": s.ragam,
            "talam": s.talam,
            "mela": s.mela,
            "artist": s.artist,
            "audio_url": s.audio_url,
            "lyrics": s.lyrics,
            "meaning": s.meaning,
            "embedding": s.embedding.tolist() if s.embedding is not None else None
        })
    
    with open("songs_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print("Saved all song metadata and embeddings to songs_metadata.json")