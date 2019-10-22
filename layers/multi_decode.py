# multi_decode.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from layers.dilated_convolution import DilatedConv2D
from layers.normalization import InstanceNormalization
from typing import Tuple, Union

import tensorflow as tf


class MultiDecode(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: Union[int, Tuple[int, int]],
                 activation_fn: type(tf.keras.layers.Layer), with_dropout=True, with_zoom=True, name=None) -> None:
        super(MultiDecode, self).__init__(trainable=True, name=name)

        # Save parameters to class.
        self.activation_fn = activation_fn
        self.filters = filters
        self.kernel_size = kernel_size
        self.with_dropout = with_dropout
        self.with_zoom = with_zoom

        # Create decode sublayers.
        if self.with_zoom:
            self.upsamp = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')
            self.upconv = tf.keras.layers.Conv2D(filters=filters, kernel_size=4, padding='same')
            self.upnorm = InstanceNormalization(axis=-1)
            if self.activation_fn:
                self.upactv = activation_fn()

        self.concatenate = tf.keras.layers.Concatenate()
        self.conv = DilatedConv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                  activation_fn=activation_fn)
        self.norm = InstanceNormalization(axis=-1)
        if self.activation_fn:
            self.actv = activation_fn()

        if with_dropout:
            self.dropout = tf.keras.layers.Dropout(rate=0.20)

    @tf.function
    def call(self, inputs, training=False) -> tf.Tensor:
        [x, y, z] = inputs

        if self.with_zoom:
            x = self.upsamp(x)
            x = self.upconv(x)
            x = self.upnorm(x)
            if self.activation_fn:
                x = self.upactv(x)

        x = self.concatenate([x, y, z])
        x = self.conv(x)
        x = self.norm(x)
        if self.activation_fn:
            x = self.actv(x)

        if self.with_dropout:
            x = self.dropout(x, training=training)

        return x
