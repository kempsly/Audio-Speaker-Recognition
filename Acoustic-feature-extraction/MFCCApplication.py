import os
import youtube_dl
import moviepy.editor as mp
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Function to download video from YouTube
# def download_youtube_video(url, output_path):
#     ydl_opts = {
#         'format': 'bestaudio/best',
#         'outtmpl': output_path + '.%(ext)s'
#     }
#     with youtube_dl.YoutubeDL(ydl_opts) as ydl:
#         ydl.download([url])

# Function to extract audio from video
# def extract_audio(video_path, output_path):
#     video = mp.VideoFileClip(video_path)
#     video.audio.write_audiofile(output_path)

# Resample the audio data
def resample_audio(input_wav_file, output_wav_file, target_sr=16000):
    y, sr = librosa.load(input_wav_file, sr=target_sr)
    librosa.output.write_wav(output_wav_file, y, sr=target_sr)

# Example usage:
input_wav_file = 'input_audio.wav'
output_wav_file = 'resampled_audio.wav'
target_sr = 16000  # Target sampling rate
resample_audio(input_wav_file, output_wav_file, target_sr)


# Function to compute MFCC features and visualize them
def visualize_mfcc(audio_path):
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

# Example usage:
# youtube_url = 'https://www.youtube.com/watch?v=_KXJ-peNr_0&list=RD_KXJ-peNr_0&start_radio=1' 
# output_video_path = 'video.mp4'
output_audio_path = 'sampleaudio.wav'

# download_youtube_video(youtube_url, output_video_path)
# extract_audio(output_video_path, output_audio_path)
visualize_mfcc(output_audio_path)

