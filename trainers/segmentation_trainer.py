# segmentation_trainer.py

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
import random as rand
import tensorflow as tf


class SegmentationTrainer(BaseTrainer):
    def __init__(self, model: BaseModel, logger: BaseLogger, train_dataset: BaseDataset,
                 valid_dataset: Optional[BaseDataset]) -> None:
        super(SegmentationTrainer, self).__init__(model, logger, train_dataset, valid_dataset)

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

        self.logger.summarize(self.model.global_step, summarizer="train", scope="model", summaries_dict={
            "total_loss": np.mean(errs)
        })

    @tf.function
    def train_step(self, data: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, ...]:
        x, y = data

        with tf.GradientTape() as tape:
            # Evaluate results on training data.
            prediction = self.model(x, training=True)
            loss = self.model.loss(y, prediction)

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

        # Output validation loss and images to TensorBoard.
        batch = rand.choice(range(len(predictions)))
        self.logger.summarize(self.model.global_step, summarizer="validation", summaries_dict={
            "prediction": np.where(np.equal(np.max(predictions[batch], axis=3, keepdims=True),
                                            predictions[batch]), 1.0, 0.0),
            "target": np.where(np.equal(np.max(targets[batch], axis=3, keepdims=True),
                                        targets[batch]), 1.0, 0.0),
        })

        # Convert lists to proper arrays.
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        # Choose highest class for predictions.
        predictions = np.argmax(predictions, axis=-1)
        targets = np.argmax(targets, axis=-1)

        # Calculate metrics
        accuracy = tf.reduce_mean(tf.cast(tf.equal(targets, predictions), dtype=tf.float32))
        iou = np.sum(np.logical_and(targets.ravel(), predictions.ravel())) / np.sum(np.logical_or(targets.ravel(), predictions.ravel()))
        precision, recall, fscore, _ = precision_recall_fscore_support(targets.ravel(), predictions.ravel())

        # Categorize validation metrics under TensorBoard.
        self.logger.summarize(self.model.global_step, summarizer="validation", scope="metrics", summaries_dict={
            "accuracy": accuracy,
            "intersection": iou
        })

        self.logger.summarize(self.model.global_step, summarizer="validation", scope="model", summaries_dict={
            "total_loss": np.mean(errs)
        })

        self.logger.summarize(self.model.global_step, summarizer="validation", scope="movable", summaries_dict={
            "precision": precision[0],
            "recall": recall[0],
            "f-score": fscore[0]
        })

        self.logger.summarize(self.model.global_step, summarizer="validation", scope="stationary", summaries_dict={
            "precision": precision[1],
            "recall": recall[1],
            "f-score": fscore[1]
        })

        self.logger.summarize(self.model.global_step, summarizer="validation", scope="water", summaries_dict={
            "precision": precision[2],
            "recall": recall[2],
            "f-score": fscore[2]
        })

    @tf.function
    def validate_step(self, data: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, ...]:
        x, y = data

        # Evaluate results on validation data.
        prediction = self.model(x, training=False)
        loss = self.model.loss(y, prediction)

        return loss, prediction, y
