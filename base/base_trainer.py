# base_trainer.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging
from base.base_dataset import BaseDataset
from base.base_logger import BaseLogger
from base.base_model import BaseModel
from typing import Optional

import abc


class BaseTrainer(abc.ABC):
    def __init__(self, model: BaseModel, logger: BaseLogger, train_dataset: BaseDataset,
                 valid_dataset: Optional[BaseDataset]) -> None:
        self.config = flags.FLAGS
        self.model = model
        self.logger = logger
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        # Build model and print summary.
        #self.model.build(input_shape=[(None,) + self.config.input_shape, (None,) + self.config.satellite_shape])
        self.model.build(input_shape=(None,) + self.config.input_shape)
        self.model.summary(print_fn=logging.info)

    def train(self) -> None:
        for _ in range(int(self.model.epoch), self.config.num_epochs + 1):
            self.train_epoch()
            if self.valid_dataset:
                self.validate_epoch()
            self.model.save_checkpoint()
            self.model.epoch.assign_add(delta=1)

    @abc.abstractmethod
    def train_epoch(self) -> None:
        pass

    @abc.abstractmethod
    def validate_epoch(self) -> None:
        raise NotImplementedError
