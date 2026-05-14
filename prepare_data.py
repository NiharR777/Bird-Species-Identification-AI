import os
import numpy as np
from tqdm import tqdm

from feature_extraction import extract_mel


DATA_PATH = "data/train_audio"

X = []
y = []

# Get all bird folders
bird_folders = os.listdir(DATA_PATH)

print("Total Species Found:", len(bird_folders))


for label, bird in enumerate(tqdm(bird_folders)):

    bird_path = os.path.join(DATA_PATH, bird)

    if not os.path.isdir(bird_path):
        continue

    files = os.listdir(bird_path)

    for file in files:

        file_path = os.path.join(bird_path, file)

        try:

            mel = extract_mel(file_path)

            if mel is not None:

                X.append(mel)
                y.append(label)

        except Exception as e:

            print("Error:", e)


# Convert into arrays
X = np.array(X)
y = np.array(y)

print("Dataset Shape:", X.shape)
print("Labels Shape:", y.shape)

# Save dataset
np.save("features/X.npy", X)
np.save("features/y.npy", y)

# Save class names
np.save("features/classes.npy", bird_folders)

print("Dataset Saved Successfully!")