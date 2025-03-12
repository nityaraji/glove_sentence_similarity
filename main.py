from glove_similarity.src.glove_loader import GloveLoader
from glove_similarity.src.sentence_embedding import SentenceEmbedding
from glove_similarity.src.similarity_calculator import SimilarityCalculator

# Load GloVe model
glove_file = r"..\glove_similarity\data\glove.6B.50d.txt"
glove_loader = GloveLoader(glove_file)

# Create SentenceEmbedding instance
sentence_embedder = SentenceEmbedding(glove_loader)

# Example sentences
sentences = {
    "sentence1": "The cat is on the mat.",
    "sentence2": "A feline is resting on the rug.",
    "sentence3": "I hate cats",
    "sentence4": "Turn on Bluetooth",
    "sentence5": "increase volume of bluetooth"
}

# Compute embeddings
embeddings = {key: sentence_embedder.get_embedding(sent) for key, sent in sentences.items()}

# Compute and print similarities
similarity_calculator = SimilarityCalculator()

print(f"Semantic Similarity of sen1 and sen2: {similarity_calculator.compute_similarity(embeddings['sentence1'], embeddings['sentence2']):.4f}")
print(f"Semantic Similarity of sen1 and sen3: {similarity_calculator.compute_similarity(embeddings['sentence1'], embeddings['sentence3']):.4f}")
print(f"Semantic Similarity of sen4 and sen5: {similarity_calculator.compute_similarity(embeddings['sentence4'], embeddings['sentence5']):.4f}")
