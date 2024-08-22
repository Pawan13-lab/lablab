import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.models import model_from_json

# Function to create and train the model
def create_and_train_model():
    # Load and preprocess the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the data using NumPy
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Define the model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

    return model

# Function to save the model architecture and weights
def save_model(model):
    # Save the model's architecture to a JSON file
    model_json = model.to_json()
    with open('model_config.json', 'w') as json_file:
        json_file.write(model_json)
    
    # Save the model's weights to an HDF5 file
    model.save_weights('model_weights.h5')

    print("Model architecture saved to model_config.json")
    print("Model weights saved to model_weights.h5")

# Function to load the model from saved files
def load_model():
    # Load the model architecture from JSON file
    with open('model_config.json', 'r') as json_file:
        model_json = json_file.read()
        model = model_from_json(model_json)

    # Load the model weights from HDF5 file
    model.load_weights('model.weights.h5')

    # Compile the model (you need to compile it to use it for evaluation or further training)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Model loaded and compiled")
    return model

# Main script execution
if __name__ == "__main__":
    # Create and train the model
    model = create_and_train_model()
    
    # Save the model
    save_model(model)
    
    # Optionally, load the model again (for demonstration)
    loaded_model = load_model()
