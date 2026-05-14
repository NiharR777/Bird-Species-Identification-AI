import librosa
import numpy as np

SR = 22050
DURATION = 5
SAMPLES_PER_TRACK = SR * DURATION


def extract_mel(file_path):

    try:
        signal, sr = librosa.load(
            file_path,
            sr=SR
        )

        # Fix audio length
        if len(signal) < SAMPLES_PER_TRACK:

            padding = SAMPLES_PER_TRACK - len(signal)

            signal = np.pad(signal, (0, padding))

        else:
            signal = signal[:SAMPLES_PER_TRACK]

        # Create Mel Spectrogram
        mel = librosa.feature.melspectrogram(
            y=signal,
            sr=sr,
            n_mels=128
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        return mel_db

    except Exception as e:

        print("Error:", e)

        return None