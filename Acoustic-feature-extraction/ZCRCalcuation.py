"""
 implementation of  zero cross rate (ZCR) in Python.
 
 Kempsly Silencieux
"""

def zero_crossing_rate(signal):
    zcr_count = 0
    for i in range(1, len(signal)):
        if (signal[i-1] >= 0 and signal[i] < 0) or (signal[i-1] < 0 and signal[i] >= 0):
            zcr_count += 1
    zcr = zcr_count / (2 * len(signal))
    return zcr

# Example usage:
signal = [0.5, -0.2, -0.3, 0.1, 0.6, -0.4, -0.7, 0.2, 0.4, -0.3]
zcr = zero_crossing_rate(signal)
print("Zero crossing rate:", zcr)
