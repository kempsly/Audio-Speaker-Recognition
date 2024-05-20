import numpy as np

def conv1d(input_sequence, kernel):
    input_len = len(input_sequence)
    kernel_len = len(kernel)
    output_len = input_len - kernel_len + 1
    output = np.zeros(output_len)
    
    for i in range(output_len):
        output[i] = np.sum(input_sequence[i:i+kernel_len] * kernel)
    
    return output

# Example usage
input_sequence = np.array([1, 2, 3, 4, 5, 6, 7])
kernel = np.array([0.5, 1, 0.5])

result = conv1d(input_sequence, kernel)
print("Convolution result:", result)
