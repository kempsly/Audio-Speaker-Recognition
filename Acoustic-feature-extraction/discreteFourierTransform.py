import cmath

def ComputeDFT(seq):
    """Compute Discrete Fourier Transform of a sequence.

    Args:
        seq: a list of floating point numbers

    Returns:
        a list of complex numbers
    """
    assert isinstance(seq, list)
    assert len(seq) > 2

    N = len(seq)
    dft_result = []
    for k in range(N):
        X_k = sum(seq[n] * cmath.exp(-2j * cmath.pi * k * n / N) for n in range(N))
        dft_result.append(X_k)
    return dft_result

# Example usage:
sequence = [1, 2, 3, 4]
dft_sequence = ComputeDFT(sequence)
print("DFT result:", dft_sequence)
