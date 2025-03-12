import numpy as np

class SentenceEmbedding:
    """Class to compute sentence embeddings using GloVe vectors."""

    def __init__(self, glove_loader, embedding_dim=50):
        self.glove_loader = glove_loader
        self.embedding_dim = embedding_dim

    def get_embedding(self, sentence):
        """Computes the mean embedding for a given sentence."""
        words = sentence.lower().split()
        valid_vectors = [self.glove_loader.get_embedding(word) for word in words if
                         self.glove_loader.get_embedding(word) is not None]

        if not valid_vectors:
            return np.zeros(self.embedding_dim)  # Return zero vector if no valid words are found

        return np.mean(valid_vectors, axis=0)