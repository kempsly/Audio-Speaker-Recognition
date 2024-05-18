    """
        Usage of the FrequencyToMel functionde finedinthe melscale.py file   
        to convert frequency from Hertz to mel.
     
       Kempsly Silencieux
    """


import math

from melScale  import FrequencyToMel

# Example frequencies in Hertz
frequencies = [100, 500, 1000, 5000, 10000]

# Convert these frequencies to the mel scale
mel_frequencies = [FrequencyToMel(freq) for freq in frequencies]

# Print the results
for freq, mel in zip(frequencies, mel_frequencies):
    print(f"Frequency: {freq} Hz -> Mel: {mel:.2f} mels")
