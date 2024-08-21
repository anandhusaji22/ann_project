import os
import numpy as np
import cv2
from keras.models import Sequential, load_model
from keras import layers
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the model and class names once globally
model = None
class_names = None

def load_images_from_folder(folder_path):
    images = []
    labels = []

    for image_file in os.listdir(folder_path):
        try:
            label, _ = image_file.split('-')
            labels.append(label)

            img_path = os.path.join(folder_path, image_file)
            img = Image.open(img_path).convert('L')  # Convert to grayscale

            img_array = np.array(img)
            _, binary_image = cv2.threshold(img_array, 10, 255, cv2.THRESH_BINARY_INV)

            images.append(binary_image)
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")

    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)

    return np.array(images), numeric_labels, label_encoder.classes_

def train_model(dataset_path='symbols'):
    print('Please wait, the network is training...!')
    images, labels, class_names = load_images_from_folder(dataset_path)
    images = images.reshape((images.shape[0], 784)).astype('float32') / 255  # Flatten and normalize
    labels = to_categorical(labels)

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    model = Sequential([
        layers.Dense(80, activation='relu', input_shape=(784,)),  # Input layer with hidden layer 1
        layers.Dense(len(class_names), activation='softmax')  # Output layer
    ])

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=8, validation_data=(x_test, y_test))

    model.save('symbol_model.h5')  # Save the trained model

    print("Model trained and saved as 'symbol_model.h5'")
    print(f"Class names: {class_names}")  # Save this for later use in prediction
    return model, class_names

def predict_image(image_path, model, class_names):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels

    img_array = np.array(img).reshape(1, 784).astype('float32') / 255
    predictions = model.predict(img_array)
    prediction = np.argmax(predictions)
    confidence = predictions[0][prediction]  # Confidence score

    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted Label: {class_names[prediction]} (Confidence: {confidence:.2f})")
    plt.show()

    return class_names[prediction], confidence
def load_trained_model():
    try:
        model = load_model('symbol_model.h5')
        _, _, class_names = load_images_from_folder('symbols')  # Ensure you load class names
        print("Loaded saved model.")
        return model, class_names
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None, None


# Main Execution Flow
if __name__ == "__main__":
    # Step 1: Train the model (only once)
    # Check if the model is already trained and saved
    if os.path.exists('symbol_model.h5'):
        model, class_names = load_trained_model()
    else:
        # Step 1: Train the model (only if not already trained)
        model, class_names = train_model()

    # Step 2: Predict using the saved model
    print('Your code is predicting. Please be cool...!')
    label, confidence = predict_image("symbol2.png", model, class_names)
    if (confidence*100)<=30:
            print("Prediction failed. Confidence is too low.")
    elif label:
        # if (confidence*100)<=30:
        #     print("Prediction failed. Confidence is too low.")
        print(f"Predicted Label: {label}, Confidence: {confidence:.2f}")
    print('Prediction completed!')
