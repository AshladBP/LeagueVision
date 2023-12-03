import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os

# Load the trained model
model = load_model(sys.argv[1])

# Path to the image you want to predict
image_path = sys.argv[2]

training_data_file = sys.argv[3]

if len(sys.argv) in [1, 2] or sys.argv[1] == "help":
    print('Usage: python3 imageRecognition.py model image training_data_file')
    exit()

# Load and preprocess the image
img = load_img(image_path, target_size=(27, 27))  # Resize to match model's expected input
img_array = img_to_array(img) / 255.0  # Normalize the image
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make a prediction
prediction = model.predict(img_array)
predicted_class_index = np.argmax(prediction, axis=1)

# Assuming class indices are the same as in the training
class_indices = os.listdir(training_data_file)  # Adjust the path as necessary
class_indices.sort()  # Ensure this matches the order during training

# Output the prediction
predicted_class = class_indices[predicted_class_index[0]]
print(f"Predicted item: {predicted_class}")
