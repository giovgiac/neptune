# triplet.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


@tf.function
def triplet_loss(anchor, positive, negative, margin=1.0):
    d_pos = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    d_neg = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    return tf.maximum(0., margin + d_pos - d_neg)


class TripletLoss(tf.keras.losses.Loss):
    def __init__(self, margin=1.0, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 name="triplet_loss"):
        super(TripletLoss, self).__init__(reduction=reduction, name=name)
        self.margin = margin

    def call(self, y_true, y_pred):
        anchor = y_true
        positive, negative = y_pred
        return triplet_loss(anchor, positive, negative, self.margin)

    def get_config(self):
        config = {
            "margin": self.margin,
        }
        base_config = super(TripletLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
