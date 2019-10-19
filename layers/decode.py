# decode.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple, Union

import tensorflow as tf


class Decode(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: Union[int, Tuple[int, int]],
                 activation_fn: type(tf.keras.layers.Layer), with_dropout=True, name=None) -> None:
        super(Decode, self).__init__(trainable=True, name=name)

        # Save parameters to class.
        self.activation_fn = activation_fn
        self.filters = filters
        self.kernel_size = kernel_size
        self.with_dropout = with_dropout

        # Create decode sublayers.
        self.upsample = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')
        self.upconv = tf.keras.layers.Conv2D(filters=filters, kernel_size=4, padding='same')
        self.conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same')
        self.activation_1 = activation_fn()
        self.conv_2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same')
        self.activation_2 = activation_fn()
        self.concatenate = tf.keras.layers.Concatenate()

        if with_dropout:
            self.dropout = tf.keras.layers.Dropout(rate=0.25)

    @tf.function
    def call(self, inputs, training=False) -> tf.Tensor:
        [x, s] = inputs

        x = self.upsample(x)
        x = self.upconv(x)
        x = self.concatenate([x, s])
        x = self.conv_1(x)
        x = self.activation_1(x)
        x = self.conv_2(x)
        x = self.activation_2(x)

        if self.with_dropout:
            x = self.dropout(x, training=training)

        return x
