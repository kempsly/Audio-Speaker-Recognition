import math

def ComputeSoftmax(vec):
    """Compute the softmax of a vector.

    Args:
        vec: a list of floating point numbers

    Returns:
        a list of floating point numbers with the same length as vec
    """
    assert isinstance(vec, list), "Input must be a list."
    assert len(vec) > 1, "Input list must have more than one element."

    # Subtract the maximum value from each element for numerical stability
    max_val = max(vec)
    exp_vec = [math.exp(i - max_val) for i in vec]
    
    # Compute the sum of exponentials
    sum_exp_vec = sum(exp_vec)
    
    # Compute the softmax values
    softmax_vec = [i / sum_exp_vec for i in exp_vec]
    
    return softmax_vec

# Example usage
input_data = [2.0, 1.0, 0.1]
print(ComputeSoftmax(input_data))
