import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import Image_Prediction as ip
import json
import tensorflow as tf
import tensorflow_hub as hub
import warnings
import logging


def JSON_file(x):
    """Checking if the given string is a valid filepath and is an .json file.
        If it's true it will pass the string back"""
    if not os.path.exists(x) or not x.endswith(".json"):
        # ArgumentTypeError if it's not an existing file or not ending with .json
        raise argparse.ArgumentTypeError("{0} does not exist or is not a json file".format(x))
    else:
        return x

def h5_file(x):
    """Checking if the given string is a valid filepath and is an .h5 model file.
    If it's true it will pass the string back"""
    if not os.path.exists(x) or not x.endswith(".h5"):
        # ArgumentTypeError if it's not an existing file or not ending with .h5
        raise argparse.ArgumentTypeError("{0} does not exist or is not a h5 file".format(x))
    else:
        return x

def file_check(x):
    """Checking if the given string is a valid filepath
        If it's true it will pass the string back"""
    if not os.path.exists(x):
        # ArgumentTypeError if it's not an existing file
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    else:
        return x

# Creating a Parser
parser = argparse.ArgumentParser(description="Model Prediction")

"""parsing the arguments in the commandline as followed:
python predict.py ./test_images/cautleya_spicata.jpg ./1634573400.h5 -cn ./label_map.json
By Default it will print the top 5 matches to the image"""

# parsing all necessary arguments
parser.add_argument("image_path", type=file_check, metavar="PATH", help="Image Path for Prediction")
parser.add_argument("model_path", type=h5_file, metavar="PATH",  help="Path of the model used as .h5 format")
parser.add_argument("-k", "--top_k", type=int, default=5, help="Get the k best predictions")
parser.add_argument("-cn", "--category_names", type=JSON_file, required=True, metavar="PATH", help="Get a JSON File which contains the category names to the numeric labels")

args = parser.parse_args()

if __name__ == "__main__":
    # Filtering all warnings. I got many cuda related errors. So I'm guessing that there are problems
    # to use the GPU. How can I fix that?
    warnings.filterwarnings('ignore')
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

    # Seperate all arguments of the argparser "container" for clarity
    image_path = args.image_path
    model_path = args.model_path
    top_k = args.top_k
    category_names = args.category_names

    # Open the json file with the class labels
    with open(category_names, 'r') as f:
        class_names = json.load(f)

    # loading the keras model
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    # Pass the image path, reloaded model and top_k to predict the flower in the image
    probs, classes = ip.predict(image_path, model, top_k=top_k)

    # printing the Label/Flowername with the corresponding probability
    for prob, x_class in zip(probs, classes):
        print(f"It's a {class_names[x_class].title()} with a Probability of {100*prob:.2f}%.")

    # Plotting the Image and Probabilities visually. Udacity's workspace couldn't display the plot
    # ip.plot_Images_and_Probability(image_path, probs, classes, class_names)
