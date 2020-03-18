# wnet.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from base.base_model import BaseModel
from layers.encode import Encode
from layers.multi_decode import MultiDecode
from typing import Optional

import tensorflow as tf


class WNet(BaseModel):
    def __init__(self, filters: int, loss: Optional[tf.keras.losses.Loss],
                 optimizer: Optional[tf.keras.optimizers.Optimizer]):
        super(WNet, self).__init__(loss, optimizer, name="W-Net")

        # Store network architecture hyperparameters.
        self.filters = filters

        # Define sublayers of the W-Net.
        self.son_encode_1 = Encode(filters=filters, kernel_size=3, activation_fn=tf.keras.layers.ELU,
                                   with_dropout=False, with_pool=False, with_reduction=False, name='sonar_encode')
        self.son_encode_2 = Encode(filters=filters * 2, kernel_size=3, activation_fn=tf.keras.layers.ELU,
                                   with_dropout=False, with_pool=False, with_reduction=True, name='sonar_encode')
        self.son_encode_3 = Encode(filters=filters * 4, kernel_size=3, activation_fn=tf.keras.layers.ELU,
                                   with_dropout=False, with_pool=False, with_reduction=True, name='sonar_encode')

        self.sat_encode_1 = Encode(filters=filters, kernel_size=3, activation_fn=tf.keras.layers.ELU,
                                   with_dropout=False, with_pool=False, with_reduction=True, name='satellite_encode')
        self.sat_encode_2 = Encode(filters=filters * 2, kernel_size=3, activation_fn=tf.keras.layers.ELU,
                                   with_dropout=False, with_pool=False, with_reduction=True, name='satellite_encode')
        self.sat_encode_3 = Encode(filters=filters * 4, kernel_size=3, activation_fn=tf.keras.layers.ELU,
                                   with_dropout=False, with_pool=False, with_reduction=True, name='satellite_encode')

        self.son_sat_cat = tf.keras.layers.Concatenate()

        self.decode_1 = MultiDecode(filters=filters * 2, kernel_size=3, activation_fn=tf.keras.layers.ELU,
                                    with_dropout=False, with_zoom=True, name='decode')
        self.decode_2 = MultiDecode(filters=filters, kernel_size=3, activation_fn=tf.keras.layers.ELU,
                                    with_dropout=False, with_zoom=True, name='decode')

        self.final = tf.keras.layers.Conv2D(filters=self.config.output_shape[2], kernel_size=1,
                                            padding='same', activation='elu')

    @tf.function
    def call(self, inputs, training=None, mask=None) -> tf.Tensor:
        [x, y] = inputs

        # Sonar
        son_e1 = self.son_encode_1(x, training=training)
        son_e2 = self.son_encode_2(son_e1, training=training)
        son_e3 = self.son_encode_3(son_e2, training=training)

        # Satellite
        sat_e1 = self.sat_encode_1(y, training=training)
        sat_e2 = self.sat_encode_2(sat_e1, training=training)
        sat_e3 = self.sat_encode_3(sat_e2, training=training)

        # Concatenate
        z = self.son_sat_cat([son_e3, sat_e3])

        # Decode
        z = self.decode_1([z, son_e2, sat_e2], training=training)
        z = self.decode_2([z, son_e1, sat_e1], training=training)

        return self.final(z)
