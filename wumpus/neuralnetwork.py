from __future__ import absolute_import, division, print_function

import hashlib

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(128,)),
    keras.layers.Dense(3, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


def string_to_int(input: str):
    """
    Parses a string to an integer list that has 128 entries
    :param input: initial string
    :return: the integer list
    """
    hash = hashlib.sha512(input.encode()).hexdigest()
    return [(ord(x) / 256) for x in hash]


def train(inp: list, output: list):
    inp = [string_to_int(i) for i in inp]
    model.fit(np.array(inp), np.array(output), epochs=2000)


def answer(question: str):
    question = string_to_int(question)
    predictions = model.predict(np.array([np.array(question)]))
    return np.argmax(predictions[0])
