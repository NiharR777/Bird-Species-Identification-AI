import os
import librosa
import matplotlib.pyplot as plt

# Path to training audio folder
DATA_PATH = "data/train_audio"

# Get first bird folder
bird_folders = os.listdir(DATA_PATH)
first_bird = bird_folders[0]

# Get first audio file
audio_files = os.listdir(os.path.join(DATA_PATH, first_bird))
first_audio = audio_files[0]

audio_path = os.path.join(DATA_PATH, first_bird, first_audio)

print("Testing file:", audio_path)

# Load audio (5 seconds)
y, sr = librosa.load(audio_path, duration=5)

print("Sample Rate:", sr)
print("Audio Length:", len(y))

# Plot waveform
plt.figure(figsize=(10,4))
plt.plot(y)
plt.title("Audio Waveform")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()
