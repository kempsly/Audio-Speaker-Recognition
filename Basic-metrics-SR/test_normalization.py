import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def t_norm(a, b):
    # Calculate the cosine similarity between vectors a and b
    similarity = cosine_similarity([a], [b])[0][0]
    
    # Apply the T-norm operation
    t_norm_result = max(0, similarity)
    
    return t_norm_result

# Example usage
vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

result = t_norm(vector_a, vector_b)
print(result)