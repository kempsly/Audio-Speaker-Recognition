from simple_1d_conv import ComputeConvolution

# Define an input signal
signal = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Define a simple kernel
kernel = [1, 0, -1]

# Compute convolution
# output[i] = sum(list(seq[i:i+len(kernel)]) * kernel)

output = ComputeConvolution(signal, kernel)

print("Input Signal:", signal)
print("Kernel:", kernel)
print("Convolution Result:", output)
