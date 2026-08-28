from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def compute_drift(vec_a: list[float], vec_b: list[float]) -> tuple[float, float]:
    """Returns (cosine_similarity, drift_score) for two embeddings."""
    # drift_score = 1 - similarity
    # takes two vectors and returns how aligned their direction is, ignoring magnitude
    sim = cosine_similarity([vec_a], [vec_b])[0][0]  # returns a 2D array; [0][0] pulls out the single number
    drift_score = 1 - sim
    return sim, drift_score
