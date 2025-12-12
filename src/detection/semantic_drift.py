from sklearn.metrics.pairwise import cosine_similarity
from .embeddings import EmbeddingExtractor


class SemanticDriftCalculator:
    def __init__(self, detector_instance):
        self.extractor = EmbeddingExtractor(detector_instance)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculates Cosine Similarity between two texts.
        Score close to 1.0 = Very Similar.
        Score close to 0.0 = High Drift / Different Meaning.
        """
        vec1 = self.extractor.get_embedding(text1)
        vec2 = self.extractor.get_embedding(text2)

        sim_score = cosine_similarity(vec1, vec2)[0][0]
        return float(sim_score)