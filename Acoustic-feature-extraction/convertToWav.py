# import moviepy.editor as mp

# def convert_mp4_to_wav(mp4_file, wav_file):
#     video = mp.VideoFileClip(mp4_file)
#     audio = video.audio
#     audio.write_audiofile(wav_file, codec='pcm_s16le')  # Set codec to ensure WAV format

# # Example usage:
# mp4_file = 'input_video.mp4'
# wav_file = 'output_audio.wav'
# convert_mp4_to_wav(mp4_file, wav_file)
import moviepy.editor as mp

def convert_mp4_audio_to_wav(mp4_audio_file, wav_file):
    audio = mp.AudioFileClip(mp4_audio_file)
    audio.write_audiofile(wav_file, codec='pcm_s16le')  # Set codec to ensure WAV format

# Example usage:
mp4_audio_file = 'sampleaudio.mp4'
wav_file = 'sampleaudio.wav'
convert_mp4_audio_to_wav(mp4_audio_file, wav_file)

