
import numpy as np
import cv2
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Specify the directory to save the images
output_dir = 'mnist_images/'

# Save a few images from the training set
for i in range(60000):  # Change the range to save more images
    image = train_images[i]
    label = train_labels[i]
    filename = f'{output_dir}mnist_{i}_label_{label}.png'
    cv2.imwrite(filename, image)

print("Images saved successfully!")