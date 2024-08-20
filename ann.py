from keras.datasets import mnist
from keras.utils import to_categorical

from keras import optimizers
from keras import metrics

import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers
from PIL import Image


def photo_add(photo):
    """Add a photo to the database."""
    




def training_data(photo):
    
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.datasets import mnist
    from keras.utils import to_categorical
    from keras import models, layers
    from PIL import Image

# Load MNIST dataset (for training purposes)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
    x_train = x_train.reshape((60000, 784)).astype('float32') / 255
    x_test = x_test.reshape((10000, 784)).astype('float32') / 255

# Convert labels to categorical (one-hot encoding)
    y_train = to_categorical(y_train)
    y = to_categorical(y_test)

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
        plt.title(f"Predicted Number: {prediction}")
        plt.show()

# Example: Predict and display a custom image
    predict_custom_image(photo)
