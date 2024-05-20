import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
audio_file_path = 'sampleaudio.wav'
y, sr = librosa.load(audio_file_path)

# Example augmentation: change the pitch
y_pitch_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=3)  # Shift pitch by 3 semitones

# Plot the original and augmented audio
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title('Original Audio')

plt.subplot(2, 1, 2)
librosa.display.waveshow(y_pitch_shifted, sr=sr)
plt.title('Pitch Shifted Audio')

plt.tight_layout()

# Create a subfolder to save plots if it doesn't exist
plot_folder = 'plots'
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

# Save the plots to the subfolder
plot_file_path = os.path.join(plot_folder, 'audio_augmentation_plot.png')
plt.savefig(plot_file_path)
print("Plot saved as:", plot_file_path)
plt.show()

# Save the augmented audio
output_file_path = 'augmented_audio.wav'
librosa.output.write_wav(output_file_path, y_pitch_shifted, sr)
print("Augmented audio saved as:", output_file_path)
