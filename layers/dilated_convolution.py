# dilated_convolution.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple, Union

import tensorflow as tf


class DilatedConv2D(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: Union[int, Tuple[int, int]], padding: str, name=None) -> None:
        super(DilatedConv2D, self).__init__(trainable=True, name=name)

        # Save parameters to class.
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding

        # Create dilated convolution sublayers.
        self.conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, dilation_rate=1, padding=padding,
                                             use_bias=False)
        self.conv_2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, dilation_rate=2, padding=padding,
                                             use_bias=False)
        self.conv_3 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, dilation_rate=4, padding=padding,
                                             use_bias=False)
        self.conv_4 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, dilation_rate=8, padding=padding,
                                             use_bias=False)
        self.conv_f = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)
        self.concat = tf.keras.layers.Concatenate()

    @tf.function
    def call(self, inputs, **kwargs) -> tf.Tensor:
        c1 = self.conv_1(inputs)
        c2 = self.conv_2(inputs)
        c3 = self.conv_3(inputs)
        c4 = self.conv_4(inputs)
        ct = self.concat([c1, c2, c3, c4])

        return self.conv_f(ct)
