import math

def ComputeCrossEntropy(p, q, epsilon=1e-12):
    """Compute the cross entropy between two distributions.

    Args:
        p: the reference, a list of floating point numbers, which is a one-hot vector
        q: the hypothesis, a list of floating point numbers, which are all non-negative and sum to 1

    Returns:
        a floating point number for the cross entropy
    """
    assert isinstance(p, list)
    assert isinstance(q, list)
    # p is a one-hot vector
    assert len(p) > 1
    assert sum(p) == 1
    assert min(p) == 0
    assert max(p) == 1
    # q is a probability distribution
    assert len(q) == len(p)
    assert min(q) >= 0
    assert math.isclose(sum(q), 1, rel_tol=1e-9)

    # Add epsilon to q to avoid log(0)
    q = [max(i, epsilon) for i in q]
    
    # Compute cross entropy
    cross_entropy = -sum([p[i] * math.log(q[i]) for i in range(len(p))])
    
    return cross_entropy

# Example usage
p = [0, 1, 0, 0]  # Reference one-hot vector
q = [0.1, 0.7, 0.1, 0.1]  # Hypothesis probability distribution

loss = ComputeCrossEntropy(p, q)
print(f"Cross entropy loss: {loss}")
