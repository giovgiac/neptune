# encode_net.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from base.base_model import BaseModel
from layers.encode import Encode
from typing import Optional, Tuple

import tensorflow as tf


class EncodeNet(BaseModel):
    def __init__(self, filters: int, loss: Optional[tf.keras.losses.Loss],
                 optimizer: Optional[tf.keras.optimizers.Optimizer]):
        # Invoke parent class constructor.
        super(EncodeNet, self).__init__(loss, optimizer, name="EncodeNet")

        # Store network architecture hyperparameters.
        self.filters = filters

        # Network layers.
        self.e1 = Encode(filters=filters * 1, kernel_size=3, activation_fn=tf.keras.layers.ReLU,
                         with_dropout=False, with_pool=True, with_reduction=False, name='encode')
        self.e2 = Encode(filters=filters * 2, kernel_size=3, activation_fn=tf.keras.layers.ReLU,
                         with_dropout=True, with_pool=True, with_reduction=False, name='encode')
        self.e3 = Encode(filters=filters * 4, kernel_size=3, activation_fn=tf.keras.layers.ReLU,
                         with_dropout=True, with_pool=True, with_reduction=False, name='encode')
        self.e4 = Encode(filters=filters * 8, kernel_size=3, activation_fn=tf.keras.layers.ReLU,
                         with_dropout=False, with_pool=True, with_reduction=False, name='encode')
        self.flat = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.drop = tf.keras.layers.Dropout(rate=0.5)
        self.d2 = tf.keras.layers.Dense(units=64, activation=None)
        self.lamb = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))

    @tf.function
    def call(self, inputs, training=None, mask=None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        y = self.e1(inputs, training=training)
        y = self.e2(y, training=training)
        y = self.e3(y, training=training)
        y = self.e4(y, training=training)
        y = self.flat(y)
        y = self.d1(y)
        y = self.drop(y)
        y = self.d2(y)
        y = self.lamb(y)

        return y
