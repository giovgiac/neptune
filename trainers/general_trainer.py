# general_trainer.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from base.base_dataset import BaseDataset
from base.base_logger import BaseLogger
from base.base_model import BaseModel
from base.base_trainer import BaseTrainer
from tqdm import tqdm
from typing import Optional
from typing import Tuple

import numpy as np
import random as rand
import tensorflow as tf


class GeneralTrainer(BaseTrainer):
    def __init__(self, model: BaseModel, logger: BaseLogger, train_dataset: BaseDataset,
                 valid_dataset: Optional[BaseDataset]) -> None:
        super(GeneralTrainer, self).__init__(model, logger, train_dataset, valid_dataset)

    def train_epoch(self) -> None:
        loop = tqdm(range(len(self.train_dataset)))
        loop.set_description("Training Epoch [{}/{}]".format(int(self.model.epoch),
                                                             self.config.num_epochs))

        errs = []
        for data, _ in zip(self.train_dataset.data, loop):
            err, grad = self.train_step(data)
            self.model.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

            # Append step data to epoch data list.
            errs.append(err)

            # Increment global step counter.
            self.model.global_step.assign_add(delta=1)

        self.logger.summarize(self.model.global_step, summarizer="train", summaries_dict={
            "total_loss": np.mean(errs)
        })

    @tf.function
    def train_step(self, data: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, ...]:
        x, y, z = data

        with tf.GradientTape() as tape:
            # Evaluate results on training data.
            prediction = self.model([x, y], training=True)
            loss = self.model.loss(z, prediction)

        grad = tape.gradient(loss, self.model.trainable_variables)
        return loss, grad

    def validate_epoch(self) -> None:
        loop = tqdm(range(len(self.valid_dataset)))
        loop.set_description("Validating Epoch {}".format(int(self.model.epoch)))

        errs = []
        predictions = []
        targets = []
        for data, _ in zip(self.valid_dataset.data, loop):
            err, prediction, target = self.validate_step(data)

            # Append step data to epoch data list.
            errs.append(err)
            predictions.append(prediction)
            targets.append(target)

        batch = rand.choice(range(len(predictions)))
        self.logger.summarize(self.model.global_step, summarizer="validation", summaries_dict={
            "prediction": predictions[batch],
            "target": targets[batch],
            "total_loss": np.mean(errs)
        })

    @tf.function
    def validate_step(self, data: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, ...]:
        x, y, z = data

        # Evaluate results on validation data.
        prediction = self.model([x, y], training=False)
        loss = self.model.loss(z, prediction)

        return loss, prediction, z
