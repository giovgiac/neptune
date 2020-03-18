# base_model.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging
from typing import Optional

import abc
import tensorflow as tf


class BaseModel(abc.ABC, tf.keras.Model):
    def __init__(self, loss: Optional[tf.keras.losses.Loss], optimizer: Optional[tf.keras.optimizers.Optimizer],
                 name: str) -> None:
        super(BaseModel, self).__init__(name=name)

        # Initialize configuration and model variables.
        self.config = flags.FLAGS
        self.loss = loss
        self.optimizer = optimizer

    @abc.abstractmethod
    def call(self, inputs, training=None, mask=None) -> tf.Tensor:
        raise NotImplementedError
