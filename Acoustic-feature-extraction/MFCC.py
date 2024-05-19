import pytube 
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

# Getting the video file from youtube & take only the audio

youtube_url = "https://www.youtube.com/watch?v=_KXJ-peNr_0&list=RD_KXJ-peNr_0&start_radio=1"

filtered_audio = pytube.YouTube(youtube_url).streams.filter(only_audio=True, file_extension="mp4")[0]
audio_downloaded = filtered_audio.download(output_path=".", filename="sampleaudio.mp4")

# ffmpeg -i sampleaudio.mp4 sampleaudio.wav   

def resample_audio(input_wav_file, output_wav_file, target_sr=16000):
    y, sr = librosa.load(input_wav_file, sr=target_sr)
    sf.write(output_wav_file, y, target_sr)

# Example usage:
input_wav_file = 'sampleaudio.wav'
output_wav_file = 'resampled_audio.wav'
target_sr = 16000  # Target sampling rate
resample_audio(input_wav_file, output_wav_file, target_sr)

# Getting simple audio characteristics
waveform, sample_rate = librosa.core.load(output_wav_file, sr=None)
print("the sample rate of the audio file is :", sample_rate)
print("----------------------------------------------------------------")

print("the waveform of the audio file is :", waveform)
print("----------------------------------------------------------------")

# Computing the mfcc feature
mfcc_features = librosa.feature.mfcc(y = waveform, sr = sample_rate)
print("the features of the audio file is :", mfcc_features)
print("----------------------------------------------------------------")

print("the shape of the audio file is :", mfcc_features.shape)


# Visualizing the features with matplotlib
plt.matshow(mfcc_features, origin='lower')
plt.show()
