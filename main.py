from preprocessing import scrape_site, convert_audio_links, clear_incomplete_songs, convert_audio_wav
from processing import embed_songs_vggish   
from network import read_data, generate_random_playlist, build_graph, sol_extraction, biased_ppr
import numpy as np

def main():
    ARCHIVE_WEBSITE = "https://www.shivkumar.org/music/index.html"
    WORKERS = 20 # For low intensity tasks
    PROCESS_INSTANTANEOUSLY = 4 # For high intensity tasks

    songs = read_data()

    playlists = []
    query_playlist = generate_random_playlist(songs)

    for i in range(10):
        playlists.append(generate_random_playlist(songs))

    G = build_graph(songs, playlists)
    Q = sol_extraction(query_playlist, G)
    recommendations = biased_ppr(Q, query_playlist, songs, playlists, 10)
    
    print("Query Playlist:")
    for song in query_playlist:
        print(f"  {song.title} - {song.composer}")
    
    query_embeddings = [song.embedding for song in query_playlist if getattr(song, 'embedding', None) is not None]
    if query_embeddings:
        query_centroid = np.mean(query_embeddings, axis=0)
    else:
        query_centroid = None

    print("\nRecommendations:")
    for song in recommendations:
        song_emb = getattr(song, 'embedding', None)
        if query_centroid is not None and song_emb is not None:
            # Calculate percent difference: ||a - b|| / ||(a + b) / 2|| * 100
            diff_norm = np.linalg.norm(query_centroid - song_emb)
            avg_norm = np.linalg.norm((query_centroid + song_emb) / 2.0)
            percent_diff = (diff_norm / avg_norm) * 100 if avg_norm > 0 else 0.0
            
            # Calculate cosine similarity
            cos_sim = np.dot(query_centroid, song_emb) / (np.linalg.norm(query_centroid) * np.linalg.norm(song_emb))
            
            print(f"  {song.title} - {song.composer}")
            print(f"    -> Percent Difference: {percent_diff:.2f}% | Cosine Sim: {cos_sim:.4f}")
        else:
            print(f"  {song.title} - {song.composer}")
            print(f"    -> No embedding available for comparison")


if __name__ == "__main__":
    main()