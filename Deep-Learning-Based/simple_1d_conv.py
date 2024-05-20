def ComputeConvolution(seq, kernel):
    """Compute 1-dim convolution.

    Args:
        seq: a list of floating point numbers as inputs
        kernel: a list of floating point numbers as the convolution kernel

    Returns:
        the output sequence, which is a list of floating point numbers
    """
    assert isinstance(seq, list)
    assert isinstance(kernel, list)
    assert len(seq) > 2
    assert len(kernel) >= 2
    assert len(seq) >= len(kernel)

    output_len = len(seq) - len(kernel) + 1
  

    output = [0.0] * output_len

    for i in range(output_len):
        output[i] = sum(seq[i:i+len(kernel)] * kernel)

    return output
