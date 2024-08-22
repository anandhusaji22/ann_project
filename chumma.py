from keras.utils import to_categorical
from keras import models, layers
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2


def load_images_from_folder(folder_path):
        images = []
        labels = []
        
       
    
        for image_file in os.listdir(folder_path):
            try:
                # Extract the label from the first part of the file name
                label, _ = image_file.split('-')

                # Convert the label to an integer (assuming the label is numeric)
                label = int(label)

                img_path = os.path.join(folder_path, image_file)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                
                # Convert PIL Image to NumPy array
                img_array = np.array(img)
                
                # Apply binary threshold using OpenCV
                _, binary_image = cv2.threshold(img_array, 10, 255, cv2.THRESH_BINARY_INV)
                
                # Display the binary image
                # plt.imshow(binary_image, cmap='gray')
                # plt.title(f"Binary Image of {image_file}")
                # plt.show()

                # Append binary image and label to lists
                images.append(binary_image)
                labels.append(label)
            except Exception as e:
                print(f"Failed to load image {img_path}: {e}")
        
        # Return the images and labels
        return np.array(images), np.array(labels)


# Specify the path to your dataset directory
dataset_path = 'symbols'  # Replace with the actual path to your dataset folder

# Load the images and labels
images, labels = load_images_from_folder(dataset_path)

# Preprocess the data
images = images.reshape((images.shape[0], 784)).astype('float32') / 255  # Flatten and normalize

# Convert labels to categorical (one-hot encoding)
labels = to_categorical(labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create Neural Network Model
model = models.Sequential()
model.add(layers.Dense(100, activation='relu'))  # Hidden layer 1
model.add(layers.Dense(50, activation='relu'))   # Hidden layer 2
model.add(layers.Dense(10, activation='sigmoid'))  # Output layer

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=40, validation_data=(x_test, y_test))

# Evaluate the model
model.evaluate(x_test, y_test)
model.evaluate(x_train, y_train)

# Function to predict and display a custom image
def predict_custom_image(image_path):
    # Load the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    
    # Preprocess the image
    img_array = np.array(img).reshape(1, 784).astype('float32') / 255
    
    # Predict the label
    prediction = np.argmax(model.predict(img_array))
    
    # Display the image and predicted label
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted Label: {prediction}")
    plt.show()

# Predict and display the image you provided
predict_custom_image("symbol4.png")

# Example usage: Pass the path to the image you want to predict
# photo_path = 'Figure_1.png'  # Replace with your image path
# training_data(photo_path)
# 