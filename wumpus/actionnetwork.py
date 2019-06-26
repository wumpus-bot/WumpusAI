from __future__ import absolute_import, division, print_function

from wumpus.util import string_to_int

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

print(tf.__version__)

model = None


def create_network(mod_count: int):
    global model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(128,)),
        keras.layers.Dense(36, activation=tf.nn.relu),
        keras.layers.Dense(mod_count, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


def train(inp: list, output: list):
    inp = [string_to_int(i) for i in inp]
    model.fit(np.array(inp), np.array(output), epochs=200)


def answer(question: str):
    question = string_to_int(question)
    predictions = model.predict(np.array([np.array(question)]))
    print(predictions[0])
    return np.argmax(predictions[0])
