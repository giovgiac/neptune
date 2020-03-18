# base_trainer.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging
from base.base_dataset import BaseDataset
from base.base_logger import BaseLogger
from base.base_model import BaseModel
from typing import Dict, Optional

import abc
import tensorflow as tf


class BaseTrainer(abc.ABC):
    def __init__(self, models: Dict[str, BaseModel], logger: BaseLogger, train_dataset: BaseDataset,
                 valid_dataset: Optional[BaseDataset]) -> None:
        self.config = flags.FLAGS
        self.models = models
        self.logger = logger
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        # TensorFlow and state variables.
        self.epoch = tf.Variable(initial_value=0, shape=(), dtype=tf.int32, trainable=False, name="epoch")
        self.global_step = tf.Variable(initial_value=0, shape=(), dtype=tf.int64, trainable=False, name="global_step")
        self.saver = tf.train.Checkpoint(epoch=self.epoch, global_step=self.global_step, **self.models)
        self.manager = tf.train.CheckpointManager(self.saver, directory=self.config.checkpoint_dir,
                                                  max_to_keep=self.config.max_to_keep)

        # Build models and print summaries.
        #self.model.build(input_shape=[(None,) + self.config.input_shape, (None,) + self.config.satellite_shape])
        for model in self.models.values():
            model.build(input_shape=(None,) + self.config.input_shape)
            model.summary(print_fn=logging.info)

    def train(self) -> None:
        for _ in range(int(self.epoch.numpy()), self.config.num_epochs + 1):
            self.train_epoch()
            if self.valid_dataset:
                self.validate_epoch()
            self.save_checkpoint()
            self.epoch.assign_add(delta=1)

    @abc.abstractmethod
    def train_epoch(self) -> None:
        pass

    @abc.abstractmethod
    def validate_epoch(self) -> None:
        raise NotImplementedError

    def load_checkpoint(self) -> None:
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=self.config.checkpoint_dir)
        if latest_checkpoint:
            logging.info("Restoring Network Checkpoint {}...".format(latest_checkpoint))
            self.saver.restore(save_path=latest_checkpoint)
            logging.info("[Epoch {}] Network Restored...".format(int(self.epoch.numpy())))

    def save_checkpoint(self) -> None:
        logging.info("[Epoch {}] Saving Network...".format(int(self.epoch.numpy())))
        self.manager.save()
        logging.info("[Epoch {}] Network Saved...".format(int(self.epoch.numpy())))
