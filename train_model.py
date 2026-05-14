import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.utils import to_categorical


# ==========================
# Load Dataset
# ==========================

X = np.load("features/X.npy")
y = np.load("features/y.npy")

print("Loaded X:", X.shape)
print("Loaded y:", y.shape)


# ==========================
# Reshape for CNN
# ==========================

X = X[..., np.newaxis]

# ==========================
# Classes
# ==========================

num_classes = len(np.unique(y))

print("Classes:", num_classes)

y = to_categorical(y, num_classes)


# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# CNN MODEL
# ==========================

model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(128, 216, 1)
    )
)

model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(
    Conv2D(
        64,
        (3, 3),
        activation='relu'
    )
)

model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(
    Conv2D(
        128,
        (3, 3),
        activation='relu'
    )
)

model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))

# ==========================
# Compile
# ==========================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================
# Train
# ==========================

history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ==========================
# Evaluate
# ==========================

loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

# ==========================
# Save Model
# ==========================

model.save("models/bird_model.h5")

print("Model Saved!")

# ==========================
# Accuracy Graph
# ==========================

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(['Train', 'Validation'])

plt.savefig("results/accuracy.png")

# ==========================
# Loss Graph
# ==========================

plt.figure()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend(['Train', 'Validation'])

plt.savefig("results/loss.png")

print("Graphs Saved!")