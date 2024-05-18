"""
Parsing the wave file using python struct library with functionnal programming. 
Preprocessings teps:
- Read the wave file
- Read the file in binary mode.
- Parse the RIFF header to get the chunk size.
- Parse the "fmt " subchunk to get the audio format, number of channels, and sampling rate.
- Parse the "data" subchunk to get the audio data.
- Calculate the length of the audio.

Kempsly SILENCIEUX
"""



import struct

def read_wav_file(file_path):
    with open(file_path, 'rb') as f:
        # Read the RIFF header
        riff = f.read(12)
        chunk_id, chunk_size, format = struct.unpack('<4sI4s', riff)
        
        # Read the "fmt " subchunk
        fmt_header = f.read(8)
        subchunk1_id, subchunk1_size = struct.unpack('<4sI', fmt_header)
        fmt_chunk = f.read(subchunk1_size)
        audio_format, num_channels, sample_rate, byte_rate, block_align, bits_per_sample = struct.unpack('<HHIIHH', fmt_chunk[:16])
        
        # Read the "data" subchunk
        data_header = f.read(8)
        subchunk2_id, subchunk2_size = struct.unpack('<4sI', data_header)
        
        # Read the actual sound data
        data = f.read(subchunk2_size)
        
        # Calculate the number of samples
        num_samples = subchunk2_size // (num_channels * (bits_per_sample // 8))
        
        # Extract the first sample
        first_sample = struct.unpack('<h', data[:2])[0]
        
        # Calculate the length of the audio in seconds
        length_in_seconds = num_samples / sample_rate
        
        return {
            "chunk_size": chunk_size,
            "num_channels": num_channels,
            "sample_rate": sample_rate,
            "first_sample_value": first_sample,
            "audio_length_seconds": length_in_seconds
        }

# Path to the WAV file
file_path = 'male_audio.wav'

# Parse the WAV file
wav_info = read_wav_file(file_path)

# Print the results
print(f"Chunk Size: {wav_info['chunk_size']}")
print(f"Number of Channels: {wav_info['num_channels']}")
print(f"Sampling Rate: {wav_info['sample_rate']}")
print(f"First Sample Value: {wav_info['first_sample_value']}")
print(f"Audio Length in Seconds: {wav_info['audio_length_seconds']}")
