import numpy as np
import tensorflow as tf

class ActionValueFunction:
    def __init__(self, input_size, output_size, alpha):
        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.models.Sequential([
            # tf.keras.layers.Dense(64, activation='relu', input_shape=(self.input_size,)),
            # tf.keras.layers.Dense(64, activation='relu'),
            # tf.keras.layers.Dense(32, activation='relu'),
            # tf.keras.layers.Dense(self.output_size, activation='linear')
            tf.keras.layers.Dense(32, input_dim=self.input_size, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.output_size, activation='linear'),
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha))
        return model

    def predict(self, x):
        return self.model.predict(x, verbose=0)
