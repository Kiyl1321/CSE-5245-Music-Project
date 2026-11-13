class Song:
    def __init__(self, title, composer, ragam, talam, mela, cat, artist, audio_url, lyrics, meaning): # Constructor
        # Basic Compositional Data
        self.title = title        # Song title
        self.composer = composer  # Author of the lyrics and music
        self.cat = cat            # Structure (Krithi, Varnam, etc.)
        
        # Musical Framework
        self.ragam = ragam        # Melodic scale 
        self.talam = talam        # Rhythmic cycle
        self.mela = mela          # Melakarta parent index 
        
        # Performance & Resources
        self.artist = artist      # Reference performer/version
        self.audio_url = audio_url    # URL to MP3 audio lesson
        self.audio_file = None    # Local path to MP3 audio lesson
        self.mp3_to_wav = False   # Whether the MP3 file has been converted to WAV

        # Embedding
        self.embedding = None     # 128-dim VGGish audio embedding (numpy array)

        # Textual Content
        self.lyrics = lyrics      # Phonetic Sahityam
        self.meaning = meaning    # English translation/context

    def __repr__(self): # Representation of the object
        emb_str = f"embedding({self.embedding.shape})" if self.embedding is not None else "None"
        return f"Song({self.title}, {self.composer}, {self.ragam}, {self.talam}, {self.mela}, {self.cat}, {self.artist}, {self.audio_url}, {self.audio_file}, {emb_str}, {self.lyrics}, {self.meaning})"

    def __eq__(self, other): # Equality check
        if not isinstance(other, Song):
            return False
        return (self.title == other.title and self.composer == other.composer 
                and self.ragam == other.ragam and self.talam == other.talam 
                and self.mela == other.mela and self.cat == other.cat 
                and self.artist == other.artist and self.audio_url == other.audio_url 
                and self.audio_file == other.audio_file 
                and self.lyrics == other.lyrics and self.meaning == other.meaning)

    def __hash__(self):
        return hash((self.title, self.composer, self.ragam, self.talam, self.mela, self.cat, self.artist, self.audio_url, self.audio_file, self.lyrics, self.meaning))