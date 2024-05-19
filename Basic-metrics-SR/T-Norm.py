import math

def cosine_similarity(vec1, vec2):
    """Compute the cosine similarity between two vectors."""
    # Compute dot product
    dot_product = sum(x * y for x, y in zip(vec1, vec2))
    
    # Compute magnitudes
    magnitude_vec1 = sum(x ** 2 for x in vec1) ** 0.5
    magnitude_vec2 = sum(y ** 2 for y in vec2) ** 0.5
    
    # Avoid division by zero
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0
    
    # Compute cosine similarity
    similarity = dot_product / (magnitude_vec1 * magnitude_vec2)
    return similarity

def t_norm(test_vector, reference_vectors):
    """Normalize the test vector using T-norm with respect to reference vectors."""
    normalized_test_vectors = []
    
    # Iterate through each reference vector
    for ref_vector in reference_vectors:
        # Compute cosine similarity between test vector and reference vector
        similarity = cosine_similarity(test_vector, ref_vector)
        
        # Normalize test vector based on cosine similarity
        normalized_test_vector = [similarity * x for x in test_vector]
        normalized_test_vectors.append(normalized_test_vector)
    
    return normalized_test_vectors

# Example usage:
test_vector = [1, 2, 3]
reference_vectors = [[4, 5, 6], [7, 8, 9], [10, 11, 12]]
normalized_test_vectors = t_norm(test_vector, reference_vectors)
print("Normalized test vectors:", normalized_test_vectors)


