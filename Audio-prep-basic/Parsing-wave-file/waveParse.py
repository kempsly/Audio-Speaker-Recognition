"""
Parsing the wave file using python struct library. 
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

f = open("male_audio.wav", 'rb')

# Getting the chunk id of the wave file
chunk_id = f.read(4)
print("The chunk id of the file :", chunk_id)

# Getting the chunk size
chunk_size = struct.unpack("<I", f.read(4))[0]
print("The chunk size of the file :", chunk_size)

# The format of the file
audio_format = f.read(4)
print("Format of the file :", audio_format)

# Subchunk reading
subchunk1_id = f.read(4)
print("The chunk id one of the file :", subchunk1_id)

# Chunk size
subchunk1_size = struct.unpack("<I", f.read(4))[0]
print("The chunk size of the file :", subchunk1_size)

# audio format
#It prints 1, that is the PCM format
audio_format = struct.unpack("<H", f.read(2))[0]
print("The audio format of the file :", audio_format)

# Getting the number of channels
num_channels = struct.unpack("<H", f.read(2))[0]
print("The number of channels of the file :", num_channels)

# Calculating the sample rate of the file
sample_rate = struct.unpack("<I", f.read(4))[0]
print("The sample rate of the audio file :", str(sample_rate) + " " + "Hz")

# Getting the byte rate of the audio file
byte_rate = struct.unpack("<I", f.read(4))[0]
print("The byte rate of the audio file :", byte_rate)
