# dizygotic_net.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from base.base_model import BaseModel
from layers.encode import Encode
from typing import Optional, Tuple

import tensorflow as tf


class DizygoticNet(BaseModel):
    def __init__(self, filters: int, loss: Optional[tf.keras.losses.Loss],
                 optimizer: Optional[tf.keras.optimizers.Optimizer]):
        # Invoke parent class constructor.
        super(DizygoticNet, self).__init__(loss, optimizer, name="DizygoticNet")

        # Store network architecture hyperparameters.
        self.filters = filters

        # Define sublayers of the Dizygotic Network.

        # Sonar encoding layers.
        self.son_e1 = Encode(filters=filters * 1, kernel_size=3, activation_fn=tf.keras.layers.ReLU,
                             with_dropout=False, with_pool=True, with_reduction=False, name='sonar_encode')
        self.son_e2 = Encode(filters=filters * 2, kernel_size=3, activation_fn=tf.keras.layers.ReLU,
                             with_dropout=False, with_pool=True, with_reduction=False, name='sonar_encode')
        self.son_e3 = Encode(filters=filters * 4, kernel_size=3, activation_fn=tf.keras.layers.ReLU,
                             with_dropout=False, with_pool=True, with_reduction=False, name='sonar_encode')
        self.son_e4 = Encode(filters=filters * 4, kernel_size=3, activation_fn=None,
                             with_dropout=False, with_pool=True, with_reduction=False, name='sonar_encode')
        self.son_f = tf.keras.layers.Flatten()

        # Satellite encoding layers.
        self.sat_e1 = Encode(filters=filters * 1, kernel_size=3, activation_fn=tf.keras.layers.ReLU,
                             with_dropout=False, with_pool=True, with_reduction=False, name='satellite_encode')
        self.sat_e2 = Encode(filters=filters * 2, kernel_size=3, activation_fn=tf.keras.layers.ReLU,
                             with_dropout=False, with_pool=True, with_reduction=False, name='satellite_encode')
        self.sat_e3 = Encode(filters=filters * 4, kernel_size=3, activation_fn=tf.keras.layers.ReLU,
                             with_dropout=False, with_pool=True, with_reduction=False, name='satellite_encode')
        self.sat_e4 = Encode(filters=filters * 4, kernel_size=3, activation_fn=None,
                             with_dropout=False, with_pool=True, with_reduction=False, name='satellite_encode')
        self.sat_f = tf.keras.layers.Flatten()

        self.son_sat_cat = tf.keras.layers.Concatenate()

        # Multilayer perceptron to match encodings.
        self.dense_1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(units=1, activation='sigmoid')

    @tf.function
    def call(self, inputs, training=None, mask=None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        [x, y] = inputs

        # Sonar
        son = self.son_e1(x, training=training)
        son = self.son_e2(son, training=training)
        son = self.son_e3(son, training=training)
        son = self.son_e4(son, training=training)
        son = self.son_f(son)

        # Satellite
        sat = self.sat_e1(y, training=training)
        sat = self.sat_e2(sat, training=training)
        sat = self.sat_e3(sat, training=training)
        sat = self.sat_e4(sat, training=training)
        sat = self.sat_f(sat)

        # Concatenate
        z = self.son_sat_cat([son, sat])

        # MLP
        z = self.dense_1(z)
        return self.dense_2(z)
