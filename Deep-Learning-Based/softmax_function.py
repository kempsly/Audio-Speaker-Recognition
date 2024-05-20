import math

def softmax(x):
    """
    Compute the softmax of a list of numbers x.
    """
    # Subtract the maximum value from each input value for numerical stability
    max_x = max(x)
    exp_x = [math.exp(i - max_x) for i in x]
    
    # Compute the sum of exponentials
    sum_exp_x = sum(exp_x)
    
    # Compute the softmax values
    softmax_x = [i / sum_exp_x for i in exp_x]
    
    return softmax_x

# Example usage
input_data = [2.0, 1.0, 0.1]
print(softmax(input_data))
