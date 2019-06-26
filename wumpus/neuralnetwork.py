from __future__ import absolute_import, division, print_function

import hashlib

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

print(tf.__version__)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(128,)),
    keras.layers.Dense(36, activation=tf.nn.relu),
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
    if len(input) < 128:
        raw_list = input + ''.join([' ' for x in range(128 - len(input))])
    elif len(input) > 128:
        # hopefully this doesn't happen
        raw_list = input[:len(input)-128]
    return [(ord(x) / 256) for x in raw_list]


def train(inp: list, output: list):
    inp = [string_to_int(i) for i in inp]
    model.fit(np.array(inp), np.array(output), epochs=200)


def answer(question: str):
    question = string_to_int(question)
    predictions = model.predict(np.array([np.array(question)]))
    print(predictions[0])
    return np.argmax(predictions[0])
