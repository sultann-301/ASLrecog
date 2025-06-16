import tensorflow as tf
import cv2
import os
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

def load_data(dataset_path):
    labels = []
    images = []
    dir_counter = 0
    for root, dirs, files in os.walk(dataset_path):
        if root == "/Users/mws/Downloads/asl_dataset":
            continue
        i = 0
        for file in files:
            if i >= 65:
                break
            if file.endswith(".jpeg"):
                file_path = os.path.join(root, file)
                curr_label = file_path[33]
                image = cv2.imread(file_path)
                image = cv2.resize(image, (400, 400))
                images.append(image)
                labels.append(ord(curr_label))
                i += 1
        dir_counter += 1        
    return labels, images

DATAPATH = "/Users/mws/Downloads/asl_dataset"
labels, images = load_data(DATAPATH)

def get_model():
    res_model = tf.keras.models.Sequential([
    # First Convolutional Layer
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(400, 400, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Second Convolutional Layer
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Third Convolutional Layer
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten Layer
    tf.keras.layers.Flatten(),

    # Fully Connected Layers
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(set(labels)), activation="softmax")
])

    res_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return res_model


unique_labels = sorted(set(labels))  # Get unique labels sorted in order
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}  # Map each ASCII value to an index
index_to_label = {idx: label for label, idx in label_to_index.items()}

# Step 2: Map the original labels to the new indices
mapped_labels = [label_to_index[label] for label in labels]

# Step 3: Convert the mapped labels to categorical
num_classes = len(unique_labels)
categorical_labels = tf.keras.utils.to_categorical(mapped_labels, num_classes=num_classes)


x_train, x_test, y_train, y_test = train_test_split(np.array(images), np.array(categorical_labels), test_size=0.2)

# model = get_model()
# model.fit(x_train, y_train, epochs=10, batch_size=32)
# model.evaluate(x_test, y_test, verbose=2)
# joblib.dump(model, 'model.pkl')