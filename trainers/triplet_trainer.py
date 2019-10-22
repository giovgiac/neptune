# triplet_trainer.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from base.base_dataset import BaseDataset
from base.base_logger import BaseLogger
from base.base_model import BaseModel
from base.base_trainer import BaseTrainer
from tqdm import tqdm
from typing import Optional
from typing import Tuple

import numpy as np
import tensorflow as tf


class TripletTrainer(BaseTrainer):
    def __init__(self, son_model: BaseModel, sat_model: BaseModel, logger: BaseLogger, train_dataset: BaseDataset,
                 valid_dataset: Optional[BaseDataset]) -> None:
        super(TripletTrainer, self).__init__(son_model, logger, train_dataset, valid_dataset)

        self.son_model = son_model
        self.sat_model = sat_model
        self.positive_mean = 0.0
        self.negative_mean = 0.0

        self.sat_model.build(input_shape=(None,) + self.config.input_shape)
        self.sat_model.summary(print_fn=logging.info)

    def train_epoch(self) -> None:
        loop = tqdm(range(len(self.train_dataset)))
        loop.set_description("Training Epoch [{}/{}]".format(int(self.model.epoch),
                                                             self.config.num_epochs))

        son_errs = []
        sat_errs = []
        positive_distances = []
        negative_distances = []
        for data, _ in zip(self.train_dataset.data, loop):
            # Apply training to the sonar encoding network.
            son_err, son_grad, son_pos, son_neg = self.sonar_train_step(data[:3])
            self.son_model.optimizer.apply_gradients(zip(son_grad, self.son_model.trainable_variables))

            # Apply training to the satellite encoding network.
            sat_err, sat_grad = self.satellite_train_step(data[3:])
            self.sat_model.optimizer.apply_gradients(zip(sat_grad, self.sat_model.trainable_variables))

            # Append step data to epoch data list.
            son_errs.append(son_err)
            sat_errs.append(sat_err)
            positive_distances.append(son_pos)
            negative_distances.append(son_neg)

            # Increment global step counter.
            self.model.global_step.assign_add(delta=1)

        # Update means
        self.positive_mean = np.mean(positive_distances)
        self.negative_mean = np.mean(negative_distances)

        self.logger.summarize(self.model.global_step, summarizer="train", scope="model", summaries_dict={
            "sonar_loss": np.mean(son_errs),
            "satellite_loss": np.mean(sat_errs)
        })

    @tf.function
    def sonar_train_step(self, data: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, ...]:
        x, y, z = data

        with tf.GradientTape() as tape:
            # Evaluate results on training data.
            anc_pred = self.son_model(x, training=True)
            pos_pred = self.sat_model(y, training=True)
            neg_pred = self.sat_model(z, training=True)

            # Calculate triplet loss.
            loss = self.son_model.loss(anc_pred, (pos_pred, neg_pred))

        grad = tape.gradient(loss, self.son_model.trainable_variables)
        return loss, grad, tf.linalg.norm(anc_pred - pos_pred), tf.linalg.norm(anc_pred - neg_pred)

    @tf.function
    def satellite_train_step(self, data: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, ...]:
        x, y, z = data

        with tf.GradientTape() as tape:
            # Evaluate results on training data.
            anc_pred = self.sat_model(x, training=True)
            pos_pred = self.son_model(y, training=True)
            neg_pred = self.son_model(z, training=True)

            # Calculate triplet loss.
            loss = self.sat_model.loss(anc_pred, (pos_pred, neg_pred))

        grad = tape.gradient(loss, self.sat_model.trainable_variables)
        return loss, grad

    def validate_epoch(self) -> None:
        loop = tqdm(range(len(self.valid_dataset)))
        loop.set_description("Validating Epoch {}".format(int(self.model.epoch)))

        son_errs = []
        sat_errs = []
        positive_distances = []
        negative_distances = []
        for data, _ in zip(self.valid_dataset.data, loop):
            son_err, son_anc, son_pos, son_neg = self.sonar_validate_step(data[:3])
            sat_err, sat_anc, sat_pos, sat_neg = self.satellite_validate_step(data[3:])

            # Append step data to epoch data list.
            son_errs.append(son_err)
            sat_errs.append(sat_err)
            positive_distances.append(tf.linalg.norm(son_anc - son_pos))
            negative_distances.append(tf.linalg.norm(son_anc - son_neg))

        # Convert lists to proper arrays.
        positive_predictions = np.hstack(positive_distances)
        negative_predictions = np.hstack(negative_distances)

        # Binarize predictions and calculate accuracy.
        positive_predictions = tf.cast(tf.less_equal(positive_predictions / (self.positive_mean + self.negative_mean), 0.5), dtype=tf.int32)
        positive_targets = tf.ones_like(positive_predictions)

        negative_predictions = tf.cast(tf.less_equal(negative_predictions / (self.positive_mean + self.negative_mean), 0.5), dtype=tf.int32)
        negative_targets = tf.zeros_like(negative_predictions)

        positive_accuracy = tf.reduce_mean(tf.cast(tf.equal(positive_targets, positive_predictions), dtype=tf.float32))
        negative_accuracy = tf.reduce_mean(tf.cast(tf.equal(negative_targets, negative_predictions), dtype=tf.float32))

        # Output validation loss to TensorBoard.
        self.logger.summarize(self.model.global_step, summarizer="validation", scope="model", summaries_dict={
            "sonar_loss": np.mean(son_errs),
            "satellite_loss": np.mean(sat_errs)
        })

        self.logger.summarize(self.model.global_step, summarizer="validation", scope="metrics", summaries_dict={
            "positive_accuracy": positive_accuracy,
            "negative_accuracy": negative_accuracy,
            "positive_distances": np.mean(positive_distances),
            "negative_distances": np.mean(negative_distances)
        })

        # Save and update epochs on the auxiliary model.
        self.sat_model.save_checkpoint()
        self.sat_model.epoch.assign_add(delta=1)

    @tf.function
    def sonar_validate_step(self, data: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, ...]:
        x, y, z = data

        # Evaluate results on validation data.
        anc_pred = self.son_model(x, training=False)
        pos_pred = self.sat_model(y, training=False)
        neg_pred = self.sat_model(z, training=False)

        # Calculate triplet loss.
        loss = self.son_model.loss(anc_pred, (pos_pred, neg_pred))
        return loss, anc_pred, pos_pred, neg_pred

    @tf.function
    def satellite_validate_step(self, data: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, ...]:
        x, y, z = data

        # Evaluate results on validation data.
        anc_pred = self.sat_model(x, training=False)
        pos_pred = self.son_model(y, training=False)
        neg_pred = self.son_model(z, training=False)

        # Calculate triplet loss.
        loss = self.sat_model.loss(anc_pred, (pos_pred, neg_pred))
        return loss, anc_pred, pos_pred, neg_pred
