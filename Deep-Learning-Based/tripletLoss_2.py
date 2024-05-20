import numpy as np

def euclidean_distance(x1, x2):
    """Compute the Euclidean distance between two vectors."""
    return np.sqrt(np.sum((np.array(x1) - np.array(x2)) ** 2))

def ComputeTripletLess(anchor, pos, neg, alpha):
    """Compute the triplet loss.

    Args:
        anchor: embedding of the anchor sample, which is a list of floating
            point numbers
        pos: embedding of the positive sample, which is a list of floating
            point numbers
        neg: embedding of the negative sample, which is a list of floating
            point numbers
        alpha: a non-negative number for the margin value

    Returns:
        the loss, which is a floating point number
    """
    assert isinstance(anchor, list)
    assert isinstance(pos, list)
    assert isinstance(neg, list)
    assert len(anchor) == len(pos)
    assert len(anchor) == len(neg)
    assert len(anchor) > 1
    assert alpha >= 0

    # Compute the Euclidean distances
    distance_ap = euclidean_distance(anchor, pos)
    distance_an = euclidean_distance(anchor, neg)
    
    # Compute the triplet loss
    loss = max(0, distance_ap - distance_an + alpha)
    
    return loss

# Example usage
anchor = [1.0, 2.0, 3.0]
pos = [1.1, 2.1, 3.1]
neg = [4.0, 5.0, 6.0]
alpha = 1.0

loss = ComputeTripletLess(anchor, pos, neg, alpha)
print(f"Triplet loss: {loss}")
