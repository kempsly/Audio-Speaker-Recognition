import math 

from test_normalization import cosine_similarity

def ComputeTNormScore(test_vec, enroll_vec, cohort_vecs): 
    """Compute the t-norm cosine score.

    Args:
        test_vec: embedding of the runtime audio, which is a list of
            floating point numbers
        enroll_vec: embedding of the enrolled speaker, which is a list of
            floating point numbers
        cohort_vec: a list of cohort embeddings, where each embedding is a list
            of floating point numbers
    
    Returns:
        a floating point number for the normalized score
    """
    assert isinstance(test_vec, list)
    assert isinstance(enroll_vec, list)
    assert isinstance(cohort_vecs, list)
    assert len(test_vec) > 1
    assert len(test_vec) == len(enroll_vec)
    assert len(cohort_vecs) > 1
    for cohort_vec in cohort_vecs:
        assert len(test_vec) == len(cohort_vec)

    # Calculate the cosine similarity between the test vector and the enroll vector
    similarity = cosine_similarity([test_vec], [enroll_vec])[0][0]

    # Calculate the T-norm score
    t_norm_score = max(0, similarity)

    return t_norm_score

# Test the ComputeTNormScore function:
# Example usage
test_vec = [1, 2, 3]
enroll_vec = [4, 5, 6]
cohort_vecs = [[7, 8, 9], [10, 11, 12]]

result = ComputeTNormScore(test_vec, enroll_vec, cohort_vecs)
print(result)
