# dilated_convolution.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from layers.normalization import InstanceNormalization
from typing import Tuple, Union

import tensorflow as tf


class DilatedConv2D(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: Union[int, Tuple[int, int]], padding: str,
                 activation_fn: type(tf.keras.layers.Layer) = tf.keras.layers.ELU, name=None) -> None:
        super(DilatedConv2D, self).__init__(trainable=True, name=name)

        # Save parameters to class.
        self.activation_fn = activation_fn
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding

        # Create dilated convolution sublayers.
        self.conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, dilation_rate=1, padding=padding,
                                             use_bias=False)
        self.norm_1 = InstanceNormalization(axis=-1)
        if self.activation_fn:
            self.actv_1 = activation_fn()

        self.conv_2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, dilation_rate=2, padding=padding,
                                             use_bias=False)
        self.norm_2 = InstanceNormalization(axis=-1)
        if self.activation_fn:
            self.actv_2 = activation_fn()

        self.conv_3 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, dilation_rate=4, padding=padding,
                                             use_bias=False)
        self.norm_3 = InstanceNormalization(axis=-1)
        if self.activation_fn:
            self.actv_3 = activation_fn()

        self.conv_4 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, dilation_rate=8, padding=padding,
                                             use_bias=False)
        self.norm_4 = InstanceNormalization(axis=-1)
        if self.activation_fn:
            self.actv_4 = activation_fn()

        self.conv_f = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding,
                                             activity_regularizer=tf.keras.regularizers.l1(0.1))
        self.concat = tf.keras.layers.Concatenate()

    @tf.function
    def call(self, inputs, **kwargs) -> tf.Tensor:
        c1 = self.conv_1(inputs)
        c1 = self.norm_1(c1)
        if self.activation_fn:
            c1 = self.actv_1(c1)

        c2 = self.conv_2(inputs)
        c2 = self.norm_2(c2)
        if self.activation_fn:
            c2 = self.actv_2(c2)

        c3 = self.conv_3(inputs)
        c3 = self.norm_3(c3)
        if self.activation_fn:
            c3 = self.actv_3(c3)

        c4 = self.conv_4(inputs)
        c4 = self.norm_4(c4)
        if self.activation_fn:
            c4 = self.actv_4(c4)

        # Concatenate all blocks and return final convolution.
        ct = self.concat([c1, c2, c3, c4])
        return self.conv_f(ct)
