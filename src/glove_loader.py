import numpy as np

class GloveLoader:
    """Class to load GloVe embeddings into a dictionary."""

    def __init__(self, file_path):
        self.file_path = file_path
        self.glove_dict = self._load_glove_model()

    def _load_glove_model(self):
        """Loads GloVe embeddings from a file into a dictionary."""
        glove_dict = {}
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], dtype="float32")
                    glove_dict[word] = vector
            print(f"Loaded {len(glove_dict)} words from GloVe.")
        except FileNotFoundError:
            print(f"Error: GloVe file '{self.file_path}' not found.")
        return glove_dict

    def get_embedding(self, word):
        """Returns the embedding of a word if present in GloVe."""
        return self.glove_dict.get(word, None)