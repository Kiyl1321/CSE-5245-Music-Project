import random
import networkx as nx
import json
import math
import numpy as np
from data_types import Song

def generate_random_playlist(songs):
    return frozenset(random.sample(songs, k=random.randrange(5, 50)))
            
def read_data():
    songs = []
    try:
        with open("songs_metadata.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                song = Song(
                    title=item.get("title"),
                    composer=item.get("composer"),
                    ragam=item.get("ragam"),
                    talam=item.get("talam"),
                    mela=item.get("mela"),
                    cat=item.get("cat"),
                    artist=item.get("artist"),
                    audio_url=item.get("audio_url"),
                    lyrics=item.get("lyrics"),
                    meaning=item.get("meaning")
                )
                if item.get("embedding") is not None:
                    song.embedding = np.array(item["embedding"])
                songs.append(song)
    except FileNotFoundError:
        print("Warning: songs_metadata.json not found. Returning empty list.")
        
    return songs

def build_graph(songs, playlists):
    G = nx.Graph()

    for song in songs:
        G.add_node(song)
    for playlist in playlists:
        G.add_node(playlist)

    labels = set()
    for song in songs:
        if song.ragam: labels.add(song.ragam)
        if song.talam: labels.add(song.talam)
        if song.composer: labels.add(song.composer)
        if song.artist: labels.add(song.artist)
        if song.mela: labels.add(song.mela)

    for label in labels:
        G.add_node(label)

    for playlist in playlists:
        for song in playlist:
            G.add_edge(playlist, song)

    for song in songs:
        if song.ragam: G.add_edge(song, song.ragam)
        if song.talam: G.add_edge(song, song.talam)
        if song.composer: G.add_edge(song, song.composer)
        if song.artist: G.add_edge(song, song.artist)
        if song.mela: G.add_edge(song, song.mela)
    
    return G

def sol_extraction(query_playlist, G):
    Vq = set()

    for song in query_playlist:
        if song in G:
            Vq.add(song) 
            for u in G.neighbors(song):
                Vq.add(u)
                for v in G.neighbors(u):
                    Vq.add(v)
    Q = G.subgraph(Vq).copy()
    return Q

def biased_ppr(Q, query_playlist, songs, playlists, k, alpha=0.4):
    song_set = set(songs)
    b_unnormalized = {}
    total_b = 0
    
    for v in Q.nodes():
        if v in song_set:
            p_v = 0.8
        elif v in playlists:
            p_v = 0.0
        else: 
            p_v = 0.2
            
        deg = Q.degree(v) + 1
        weight = p_v * math.log(deg) if deg > 0 else 0
        
        b_unnormalized[v] = weight
        total_b += weight
        
    personalization = {}
    if total_b > 0:
        personalization = {v: (weight / total_b) for v, weight in b_unnormalized.items()}
    else:
        personalization = {v: 1/len(Q) for v in Q.nodes()}

    scores = nx.pagerank(Q, alpha=alpha, personalization=personalization)
    
    recommendations = []
    query_set = set(query_playlist)
    
    for node, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if node in song_set and node not in query_set:
            recommendations.append(node)
            if len(recommendations) >= k:
                break
                
    return recommendations