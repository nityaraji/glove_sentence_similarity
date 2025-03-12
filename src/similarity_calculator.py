from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimilarityCalculator:
    """Class to compute similarity between sentence embeddings."""

    @staticmethod
    def compute_similarity(embedding1, embedding2):
        """Computes cosine similarity between two embeddings."""
        if np.linalg.norm(embedding1) == 0 or np.linalg.norm(embedding2) == 0:
            return 0.0  # Return 0 similarity if one of the vectors is all zeros

        return cosine_similarity([embedding1], [embedding2])[0][0]