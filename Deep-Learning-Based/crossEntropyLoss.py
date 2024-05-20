import numpy as np

def compute_cross_entropy_loss(p, q, epsilon=1e-12):
    """
    Compute the cross entropy loss between the reference (one-hot) vector p
    and the hypothesis (probability distribution) vector q.

    Args:
        p: list or numpy array of shape (N,), one-hot encoded vector
        q: list or numpy array of shape (N,), probability distribution vector
        epsilon: small value to avoid log(0)

    Returns:
        float: cross entropy loss
    """
    # Convert to numpy arrays if they are not already
    p = np.array(p)
    q = np.array(q)
    
    # Ensure q is a valid probability distribution
    q = np.clip(q, epsilon, 1. - epsilon)
    
    # Compute cross entropy loss
    cross_entropy_loss = -np.sum(p * np.log(q))
    
    return cross_entropy_loss

# Example usage
p = [0, 1, 0, 0]  # Reference one-hot vector
q = [0.1, 0.7, 0.1, 0.1]  # Hypothesis probability distribution

loss = compute_cross_entropy_loss(p, q)
print(f"Cross entropy loss: {loss}")
