# base_evaluator.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging
from base.base_dataset import BaseDataset
from base.base_model import BaseModel
from typing import Dict

import abc
import tensorflow as tf


class BaseEvaluator(abc.ABC):
    def __init__(self, models: Dict[str, BaseModel], dataset: BaseDataset) -> None:
        self.config = flags.FLAGS
        self.models = models
        self.dataset = dataset

        # TensorFlow and state variables.
        self.saver = tf.train.Checkpoint(**self.models)
        self.manager = tf.train.CheckpointManager(self.saver, directory=self.config.checkpoint_dir,
                                                  max_to_keep=self.config.max_to_keep)

    @abc.abstractmethod
    def evaluate(self) -> None:
        raise NotImplementedError

    def load_checkpoint(self) -> None:
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=self.config.checkpoint_dir)
        if latest_checkpoint:
            logging.info("Restoring Network Checkpoint {}...".format(latest_checkpoint))
            self.saver.restore(save_path=latest_checkpoint)
            logging.info("Network Restored...")
