"""
    Frequency in Hertz to Mel   
    The mel scale is a perceptual scale of pitches judged by listeners 
    to be equal in distance from one another. The formula used for converting a frequency 
    f in Hertz to the mel scale is: mel =2595 * log10 * (1+ 𝑓/700)

    
"""

import math

def FrequencyToMel(freq):
    """Convert frequency to mel.

    Args:
        freq: the frequency in Hertz, a positive floating point number

    Returns:
        a positive floating point number
    """
    assert freq > 0, "Frequency must be a positive number"
    
    mel = 2595 * math.log10(1 + freq / 700)
    return mel


