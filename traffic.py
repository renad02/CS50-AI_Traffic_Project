import cv2                                                                      # OpenCV, used to read and resize images
import numpy as np                                                              # for numerical operations
import os                                                                       # to navigate through folders
import sys                                                                      # to handle command-line arguments and errors
import tensorflow as tf                                                         # for building and training the neural network

from sklearn.model_selection import train_test_split                            # splits data into training/testing parts

EPOCHS = 10                                                                     # How many times the model will train over all data
IMG_WIDTH = 30                                                                  # Every image is resized to 30×30 pixels
IMG_HEIGHT = 30
NUM_CATEGORIES = 43                                                             # Total number of traffic sign classes
TEST_SIZE = 0.4                                                                 # 40% of data will be used for testing


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")          # to also save the trained model

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])                                     # Calls load_data() to read all the images and their labels from the folder.

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)                              # Converts label integers (like 0, 1, 2, …) into one-hot encoded vectors.
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE                 # Converts lists into NumPy arrays (required by TensorFlow).
    )

    # Get a compiled neural network
    model = get_model()                                                         # This calls another function get_model() that builds the neural network architecture.

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)                                  # Runs the forward pass + backpropagation for 10 epochs. Each epoch, the model adjusts its weights to reduce the loss.

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)                                  # Tests how well the trained model performs on unseen data (x_test). Prints the loss and accuracy.

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")                                    # If the user provides a filename like model.h5, the trained model is saved for future use.


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    
    print(f'Loading images from dataset in directory "{data_dir}"')             # This function reads all the images from subfolders and labels them properly.

    images = []
    labels = []

    # Iterate through sign folders in directory:
    for foldername in os.listdir(data_dir):                                     # Each folder corresponds to one category (e.g., "0", "1", … "42").
        # Error Checking Data Folder
        try:                                                                    
            int(foldername)
        except ValueError:
            print("Warning! Non-integer folder name in data directory! Skipping...")
            continue                                                            # Skips any non-numeric folders (for example, if there’s a README file).
    # Iterate through images in each folder
        for filename in os.listdir(os.path.join(data_dir, foldername)):         # Loop through images inside each folder        
            # Open each image and resize to be IMG_WIDTH X IMG HEIGHT
            img = cv2.imread(os.path.join(data_dir, foldername, filename))
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            # Normalizes pixel values (from 0–255 → 0–1):
            img=img/255

            # Append Resized Image and its label to lists
            images.append(img)
            labels.append(int(foldername))

    # Check number of Images Matches Number of Labels:
    if len(images) != len(labels):                                              # Ensures that each image has a corresponding label.
        sys.exit('Error when loading data, number of images did not match number of labels!')
    else:
        print(f'{len(images)}, {len(labels)} labelled images loaded successfully from dataset!')

    return (images, labels)


def get_model():                                                                # Creates and compiles a Convolutional Neural Network using Keras.
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Start a Sequential model. Sequential = layers stacked one after another.
    model = tf.keras.models.Sequential([

    # Add 2 sequential 64 filter, 3x3 Convolutional Layers Followed by 2x2 Pooling
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)), # Detects patterns/features in the image (edges, shapes) using filters. activation="relu" Makes model learn non-linear features.
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),                             # Reduces image size and keeps important information. 
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),                             # So these two Conv+Pool pairs extract spatial features from the images.

    # Flatten layers
    tf.keras.layers.Flatten(),                                                  # converts 2D feature maps into a 1D vector.

    # Add A Dense Hidden layer with 512 units and 50% dropout
    tf.keras.layers.Dense(512, activation="relu"),                              # adds a fully connected layer with 512 neurons. 512 = the number units in this layer.
    tf.keras.layers.Dropout(0.5),                                               # randomly turns off 50% of neurons during training (reduces overfitting).

    # Add Dense Output layer with 43 output units
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")                 # 43 neurons → one per traffic sign category. softmax converts the outputs into probabilities that sum to 1.
    ])

    # Set additional model settings and compile:
    model.compile(optimizer='adam',                                             # adam: optimizer for updating weights efficiently
              loss='categorical_crossentropy',                                  # loss function for multi-class classification
              metrics=['accuracy'])                                             # performance metric

    # Return model for training and testing
    return model


if __name__ == "__main__":
    main()
