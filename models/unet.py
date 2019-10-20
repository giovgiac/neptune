# unet.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from base.base_model import BaseModel
from layers.decode import Decode
from layers.encode import Encode
from typing import Optional

import tensorflow as tf


class UNet(BaseModel):
    def __init__(self, filters: int, loss: Optional[tf.keras.losses.Loss],
                 optimizer: Optional[tf.keras.optimizers.Optimizer]):
        super(UNet, self).__init__(loss, optimizer, name="U-Net")

        # Store network architecture hyperparameters.
        self.filters = filters

        # Define sublayers of the U-Net.
        self.encode_1 = Encode(filters=filters, kernel_size=3, activation_fn=tf.keras.layers.ELU,
                               with_pool=False, with_reduction=False, name='encode')
        self.encode_2 = Encode(filters=filters * 2, kernel_size=3, activation_fn=tf.keras.layers.ELU,
                               with_pool=True, with_reduction=False, name='encode')
        self.encode_3 = Encode(filters=filters * 4, kernel_size=3, activation_fn=tf.keras.layers.ELU,
                               with_pool=True, with_reduction=False, name='encode')
        self.encode_4 = Encode(filters=filters * 8, kernel_size=3, activation_fn=tf.keras.layers.ELU,
                               with_pool=True, with_reduction=False, name='encode')
        self.encode_5 = Encode(filters=filters * 16, kernel_size=3, activation_fn=tf.keras.layers.ELU,
                               with_pool=True, with_reduction=False, name='encode')

        self.decode_1 = Decode(filters=filters * 8, kernel_size=3, activation_fn=tf.keras.layers.ELU,
                               with_dropout=True, with_zoom=True, name='decode')
        self.decode_2 = Decode(filters=filters * 4, kernel_size=3, activation_fn=tf.keras.layers.ELU,
                               with_dropout=True, with_zoom=True, name='decode')
        self.decode_3 = Decode(filters=filters * 2, kernel_size=3, activation_fn=tf.keras.layers.ELU,
                               with_dropout=False, with_zoom=True, name='decode')
        self.decode_4 = Decode(filters=filters, kernel_size=3, activation_fn=tf.keras.layers.ELU,
                               with_dropout=False, with_zoom=True, name='decode')

        self.final = tf.keras.layers.Conv2D(filters=self.config.output_shape[2], kernel_size=1,
                                            padding='same', activation='softmax')

    def call(self, inputs, training=None, mask=None) -> tf.Tensor:
        e1 = self.encode_1(inputs)
        e2 = self.encode_2(e1, training=training)
        e3 = self.encode_3(e2, training=training)
        e4 = self.encode_4(e3, training=training)
        x = self.encode_5(e4, training=training)
        x = self.decode_1([x, e4], training=training)
        x = self.decode_2([x, e3], training=training)
        x = self.decode_3([x, e2], training=training)
        x = self.decode_4([x, e1], training=training)

        return self.final(x)
