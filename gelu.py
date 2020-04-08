import tensorflow as tf
import numpy as np


class GeLU(tf.keras.layers.Layer):
    gamma = 0.044715

    def __init__(self):
        super(GeLU, self).__init__()

    def call(self, inputs, **kwargs):
        y = 0.5 * inputs + tf.tanh(tf.sqrt(2 / np.pi) * (inputs + self.gamma * tf.pow(inputs, 3)))
        return y
