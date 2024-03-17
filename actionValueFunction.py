import numpy as np
import tensorflow as tf

class NeuralNetwork:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.input_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.output_size, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict(self, x):
        return self.model.predict(x)
