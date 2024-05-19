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

def ComputeCosine(vec1, vec2):
    """Compute the cosine similarity between two vectors.
    
    Args:
        vec1: a list of floating point numbers
        vec2: a list of floating point numbers with the same length as vec1
        
    Returns:
        a floating point number for the cosine similarity
    """
    assert isinstance(vec1, list)
    assert isinstance(vec2, list)
    assert len(vec1) > 1
    assert len(vec1) == len(vec2)
    
    return cosine_similarity(vec1, vec2)
