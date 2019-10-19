# base_dataset.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from typing import Dict

import abc
import tensorflow as tf


class BaseLogger(abc.ABC):
    def __init__(self) -> None:
        self.config = flags.FLAGS

    @abc.abstractmethod
    def summarize(self, step: tf.Variable, summarizer="train", scope="", summaries_dict: Dict = None) -> None:
        raise NotImplementedError
