# encode.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from layers.dilated_convolution import DilatedConv2D
from typing import Tuple, Union

import tensorflow as tf


class Encode(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: Union[int, Tuple[int, int]],
                 activation_fn: type(tf.keras.layers.Layer), with_pool=False, with_reduction=True, name=None) -> None:
        super(Encode, self).__init__(trainable=True, name=name)

        # Save parameters to class.
        self.activation_fn = activation_fn
        self.filters = filters
        self.kernel_size = kernel_size
        self.with_pool = with_pool
        self.with_reduction = with_reduction

        # Create encode sublayers.
        self.conv_1 = DilatedConv2D(filters=filters, kernel_size=kernel_size, padding='same')
        self.actv_1 = activation_fn()

        if self.with_pool:
            self.pool = tf.keras.layers.MaxPool2D(pool_size=2, padding='same')

        if self.with_reduction:
            self.conv_2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=2, padding='same')
            self.actv_2 = activation_fn()

    @tf.function
    def call(self, inputs, training=False) -> tf.Tensor:
        x = self.conv_1(inputs)
        x = self.actv_1(x)

        if self.with_pool:
            x = self.pool(x)

        if self.with_reduction:
            x = self.conv_2(x)
            x = self.actv_2(x)

        return x
