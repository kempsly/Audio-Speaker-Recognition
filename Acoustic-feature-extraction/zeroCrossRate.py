def ComputeZeroCross(signal):
    """Compute the zero cross rate of the signal.

    Args:
        signal: a list of floating point numbers

    Returns:
        an integer for the number of zero crossings in the signal
    """
    assert isinstance(signal, list)
    assert len(signal) > 1

    zero_crossings = 0
    for i in range(1, len(signal)):
        if (signal[i-1] >= 0 and signal[i] < 0) or (signal[i-1] < 0 and signal[i] >= 0):
            zero_crossings += 1

    return zero_crossings

# Example usage:
signal = [0.5, -0.2, -0.3, 0.1, 0.6, -0.4, -0.7, 0.2, 0.4, -0.3]
zcr = ComputeZeroCross(signal)
print("Zero crossings:", zcr)
