"""
    Implementation of the metric cosine similarity function
    
    Kempsly S.
"""

import math

def dot_product(vec1, vec2):
    return sum(x * y for x, y in zip(vec1, vec2))

def vector_magnitude(vec):
    return math.sqrt(sum(x ** 2 for x in vec))

def cosine_similarity(vec1, vec2):
    dot_prod = dot_product(vec1, vec2)
    mag1 = vector_magnitude(vec1)
    mag2 = vector_magnitude(vec2)
    
    if mag1 == 0 or mag2 == 0:
        return 0  # Handle division by zero
    
    return dot_prod / (mag1 * mag2)

# Example usage:
vector1 = [3, 4, 5]
vector2 = [1, 2, 3]

similarity = cosine_similarity(vector1, vector2)
print("Cosine similarity:", similarity)
