# Import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def process_image(image_numpy, image_size=224):
    """ Resizes the shape of the image and normalizes the pixels
    Returns an image as a numpy array"""
    image = tf.image.resize(image_numpy, (image_size, image_size))
    image /= 255
    return image.numpy()

def predict(image_path, model, top_k=5):
    """ Function to predict the given image with the given model. By default
    it will display the best 5 matches to image.
    Returns the Probability to the corresponding classes"""
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    prediction = prediction.reshape(102)

    labels = np.argsort(prediction)[-top_k:][::-1]
    probs = prediction[labels]
    classes = [str(label + 1) for label in labels]

    return probs, classes

def plot_Images_and_Probability(image_path, probs, classes, class_names=False):
    """Plotting the image with the corresponding probabilities as a horizontal bar chart"""

    im = Image.open(image_path)
    test_image = np.asarray(im)

    processed_test_image = process_image(test_image)
    names = list()
    if class_names:
        for x_class in classes:
            names.append(class_names[x_class])
    else:
        names = classes

    plt.rcdefaults()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(processed_test_image)
    ax1.axis("off")
    ax2.barh(names, probs * 100)
    ax2.set_title('Class Probability')
    ax2.set_xlabel("Probability in Percent")
    ax2.set_xlim([0, 100])
    plt.tight_layout()
    plt.show()