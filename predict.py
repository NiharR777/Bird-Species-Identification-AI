import os
import numpy as np
import librosa

from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

from feature_extraction import extract_mel


# Paths
MODEL_PATH = "models/bird_model.h5"
CLASSES_PATH = "features/classes.npy"
DATA_PATH = "data/train_audio"


# Load trained model
print("Loading model...")
model = load_model(MODEL_PATH)

# Load class labels
classes = np.load(CLASSES_PATH, allow_pickle=True)

print("Model loaded!")
print("Classes:", classes)


def predict_bird(audio_path):
    """
    Predict bird species from audio file
    """

    # Extract features
    mel = extract_mel(audio_path)

    # Add batch & channel dimension
    mel = mel[np.newaxis, ..., np.newaxis]

    # Predict
    pred = model.predict(mel)

    # Get best class
    index = np.argmax(pred)

    bird_code = classes[index]
    confidence = np.max(pred) * 100

    return bird_code, confidence


# ---------------- MAIN ---------------- #

if __name__ == "__main__":

    # Pick one test audio automatically
    first_bird = os.listdir(DATA_PATH)[0]
    first_file = os.listdir(os.path.join(DATA_PATH, first_bird))[0]

    test_audio = os.path.join(DATA_PATH, first_bird, first_file)

    print("\nTesting on:", test_audio)

    bird, conf = predict_bird(test_audio)

    print("\nPrediction Result")
    print("------------------")
    print("Bird Species Code:", bird)
    print("Confidence:", round(conf, 2), "%")
