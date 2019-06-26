from __future__ import absolute_import, division, print_function

from wumpus.util import string_to_int

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

print(tf.__version__)

model: keras.Sequential = None


def create_network():
    global model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(4, 128)),
        keras.layers.Dense(29, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


def train(inp: list, output: list):
    """
    Trains the nn to
    :param inp: list of lists of strings : list of lists of (4 questions from data set -> string) [["What", "When", ...], ["Who", "Why", ...], ...]
    :param output: list of argument indexes : every entry of inp has 1 arg indexe for the first element (first question from data set)
    """
    inp = [[string_to_int(j) for j in i] for i in inp]
    model.fit(np.array(inp), np.array(output), epochs=200)


def get_arg_index(question: str, premades: list):
    """

    :param question: The user input
    :param premades: 3 questions from the dataset
    :return: Index of argument
    """
    question = string_to_int(question)
    premades = [string_to_int(premade) for premade in premades]
    predictions = model.predict(np.array([np.array(question), premades[0], premades[1], premades[3]]))
    print(predictions[0])
    return np.argmax(predictions[0])
