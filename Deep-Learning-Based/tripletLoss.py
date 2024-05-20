import numpy as np

def euclidean_distance(x1, x2):
    """Compute the Euclidean distance between two vectors."""
    return np.sqrt(np.sum((x1 - x2) ** 2))

def triplet_loss(anchor, positive, negative, margin=1.0):
    """Compute the triplet loss.

    Args:
        anchor: numpy array, the anchor sample.
        positive: numpy array, the positive sample (similar to the anchor).
        negative: numpy array, the negative sample (different from the anchor).
        margin: float, the margin value.

    Returns:
        float, the triplet loss value.
    """
    distance_ap = euclidean_distance(anchor, positive)
    distance_an = euclidean_distance(anchor, negative)
    
    loss = np.maximum(0, distance_ap - distance_an + margin)
    
    return loss

# Example usage
anchor = np.array([1.0, 2.0, 3.0])
positive = np.array([1.1, 2.1, 3.1])
negative = np.array([4.0, 5.0, 6.0])

loss = triplet_loss(anchor, positive, negative)
print(f"Triplet loss: {loss}")
