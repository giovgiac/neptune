# matching_trainer.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from base.base_dataset import BaseDataset
from base.base_logger import BaseLogger
from base.base_model import BaseModel
from base.base_trainer import BaseTrainer
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from typing import Optional
from typing import Tuple

import numpy as np
import tensorflow as tf


class MatchingTrainer(BaseTrainer):
    def __init__(self, model: BaseModel, logger: BaseLogger, train_dataset: BaseDataset,
                 valid_dataset: Optional[BaseDataset]) -> None:
        super(MatchingTrainer, self).__init__({"model": model}, logger, train_dataset, valid_dataset)

        # Neural network model references.
        self.model = model

    def train_epoch(self) -> None:
        loop = tqdm(range(len(self.train_dataset)))
        loop.set_description("Training Epoch [{}/{}]".format(int(self.epoch), self.config.num_epochs))

        errs = []
        predictions = []
        targets = []
        for data, _ in zip(self.train_dataset.data, loop):
            err, grad, prediction, target = self.train_step(data)
            self.model.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

            # Append step data to epoch data list.
            errs.append(err)
            predictions.append(prediction)
            targets.append(target)

            # Increment global step counter.
            self.global_step.assign_add(delta=1)

        # Convert lists to proper arrays.
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        # Binarize predictions and targets based on threshold.
        predictions = tf.cast(tf.less_equal(predictions, 0.5), dtype=tf.int32)
        targets = tf.cast(tf.less_equal(targets, 0.5), dtype=tf.int32)

        # Calculate metrics on validation data.
        accuracy = tf.reduce_mean(tf.cast(tf.equal(targets, predictions), dtype=tf.float32))

        self.logger.summarize(self.global_step, summarizer="train", scope="model", summaries_dict={
            "accuracy": accuracy,
            "total_loss": np.mean(errs)
        })

    @tf.function
    def train_step(self, data: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, ...]:
        x, y, z = data
        z = tf.expand_dims(z, axis=-1)

        with tf.GradientTape() as tape:
            # Evaluate results on training data.
            prediction = self.model([x, y], training=True)
            loss = self.model.loss(z, prediction)

        grad = tape.gradient(loss, self.model.trainable_variables)
        return loss, grad, prediction, z

    def validate_epoch(self) -> None:
        loop = tqdm(range(len(self.valid_dataset)))
        loop.set_description("Validating Epoch {}".format(int(self.epoch)))

        errs = []
        predictions = []
        targets = []
        for data, _ in zip(self.valid_dataset.data, loop):
            err, prediction, target = self.validate_step(data)

            # Append step data to epoch data list.
            errs.append(err)
            predictions.append(prediction)
            targets.append(target)

        # Convert lists to proper arrays.
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        # Binarize predictions and targets based on threshold.
        predictions = tf.cast(tf.less_equal(predictions, 0.5), dtype=tf.int32)
        targets = tf.cast(tf.less_equal(targets, 0.5), dtype=tf.int32)

        # Calculate metrics on validation data.
        accuracy = tf.reduce_mean(tf.cast(tf.equal(targets, predictions), dtype=tf.float32))
        precision, recall, fscore, _ = precision_recall_fscore_support(targets, predictions)

        # Output validation loss to TensorBoard.
        self.logger.summarize(self.global_step, summarizer="validation", scope="model", summaries_dict={
            "accuracy": accuracy,
            "total_loss": np.mean(errs)
        })

        # Categorize validation metrics under TensorBoard.
        self.logger.summarize(self.global_step, summarizer="validation", scope="match", summaries_dict={
            "precision": precision[1],
            "recall": recall[1],
            "f-score": fscore[1]
        })

        self.logger.summarize(self.global_step, summarizer="validation", scope="mismatch", summaries_dict={
            "precision": precision[0],
            "recall": recall[0],
            "f-score": fscore[0]
        })

    @tf.function
    def validate_step(self, data: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, ...]:
        x, y, z = data
        z = tf.expand_dims(z, axis=-1)

        # Evaluate results on validation data.
        prediction = self.model([x, y], training=False)
        loss = self.model.loss(z, prediction)

        return loss, prediction, z
