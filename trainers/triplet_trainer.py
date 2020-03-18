# triplet_trainer.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
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


class TripletTrainer(BaseTrainer):
    def __init__(self, son_model: BaseModel, sat_model: BaseModel, logger: BaseLogger, train_dataset: BaseDataset,
                 valid_dataset: Optional[BaseDataset]) -> None:
        super(TripletTrainer, self).__init__(son_model, logger, train_dataset, valid_dataset)

        # Neural network model references.
        self.son_model = son_model
        self.sat_model = sat_model

        # Parameters for matching probability distributions.
        self.positives = {"mean": None, "std": None}
        self.negatives = {"mean": None, "std": None}

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
            son_err, son_grad, son_pos, son_neg = self.sonar_train_step(data)
            self.son_model.optimizer.apply_gradients(zip(son_grad, self.son_model.trainable_variables))

            # Apply training to the satellite encoding network.
            sat_err, sat_grad = self.satellite_train_step(data)
            self.sat_model.optimizer.apply_gradients(zip(sat_grad, self.sat_model.trainable_variables))

            # Append step data to epoch data list.
            son_errs.append(son_err)
            sat_errs.append(sat_err)
            positive_distances.append(son_pos)
            negative_distances.append(son_neg)

            # Increment global step counter.
            self.model.global_step.assign_add(delta=1)

        # Turn lists into proper numpy arrays.
        positive_distances = np.hstack(positive_distances)
        negative_distances = np.hstack(negative_distances)

        # Update means and deviations.
        self.positives["mean"] = np.mean(positive_distances)
        self.positives["std"] = np.std(positive_distances)
        self.negatives["mean"] = np.mean(negative_distances)
        self.negatives["std"] = np.std(negative_distances)

        self.logger.summarize(self.model.global_step, summarizer="train", scope="model", summaries_dict={
            "sonar_loss": np.mean(son_errs),
            "satellite_loss": np.mean(sat_errs)
        })

        self.logger.summarize(self.model.global_step, summarizer="train", scope="distribution", summaries_dict={
            "positive_mean": self.positives["mean"],
            "positive_std": self.positives["std"],
            "negative_mean": self.negatives["mean"],
            "negative_std": self.negatives["std"]
        })

        """"
        CODE FOR: MULTIVARIATE GAUSSIAN MATCHING PROBABILITY

        # Update means and covariances
        self.positives["mean"] = np.mean(positive_distances, axis=0)
        self.positives["std"] = np.cov(np.transpose(positive_distances))
        self.negatives["mean"] = np.mean(negative_distances, axis=0)
        self.negatives["std"] = np.cov(np.transpose(negative_distances))

        self.logger.summarize(self.model.global_step, summarizer="train", scope="distribution", summaries_dict={
            "positive_mean": np.linalg.norm(self.positives["mean"]),
            "positive_std": np.linalg.norm(self.positives["std"]),
            "negative_mean": np.linalg.norm(self.negatives["mean"]),
            "negative_std": np.linalg.norm(self.negatives["std"])
        })
        """

    @tf.function
    def sonar_train_step(self, data: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, ...]:
        x, y, _, z = data

        with tf.GradientTape() as tape:
            # Evaluate results on training data.
            anc_pred = self.son_model(x, training=True)
            pos_pred = self.sat_model(y, training=True)
            neg_pred = self.sat_model(z, training=True)

            # Calculate triplet loss.
            loss = self.son_model.loss(anc_pred, (pos_pred, neg_pred))

        grad = tape.gradient(loss, self.son_model.trainable_variables)
        return loss, grad, tf.linalg.norm(anc_pred - pos_pred, axis=1), tf.linalg.norm(anc_pred - neg_pred, axis=1)

    @tf.function
    def satellite_train_step(self, data: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, ...]:
        y, x, z, _ = data

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
            son_err, son_anc, son_pos, son_neg = self.sonar_validate_step(data)
            sat_err, sat_anc, sat_pos, sat_neg = self.satellite_validate_step(data)

            # Append step data to epoch data list.
            son_errs.append(son_err)
            sat_errs.append(sat_err)
            positive_distances.append(son_pos)
            negative_distances.append(son_neg)

        # Turn lists into proper numpy arrays.
        positive_distances = np.hstack(positive_distances)
        negative_distances = np.hstack(negative_distances)

        # Calculate matching probabilities.
        positive_probabilities = 1 / (1 + self.positives["std"] / self.negatives["std"] * np.exp(0.5 * np.power((positive_distances - self.positives["mean"]) / self.positives["std"], 2) - 0.5 * np.power((positive_distances - self.negatives["mean"]) / self.negatives["std"], 2)))
        negative_probabilities = 1 / (1 + self.positives["std"] / self.negatives["std"] * np.exp(0.5 * np.power((negative_distances - self.positives["mean"]) / self.positives["std"], 2) - 0.5 * np.power((negative_distances - self.negatives["mean"]) / self.negatives["std"], 2)))

        # Binarize predictions and calculate accuracy.
        positive_predictions = tf.cast(tf.greater_equal(positive_probabilities, 0.5), dtype=tf.int32)
        positive_targets = tf.ones_like(positive_predictions)

        negative_predictions = tf.cast(tf.greater_equal(negative_probabilities, 0.5), dtype=tf.int32)
        negative_targets = tf.zeros_like(negative_predictions)

        positive_accuracy = tf.reduce_mean(tf.cast(tf.equal(positive_targets, positive_predictions), dtype=tf.float32))
        negative_accuracy = tf.reduce_mean(tf.cast(tf.equal(negative_targets, negative_predictions), dtype=tf.float32))

        # Output validation loss to TensorBoard.
        self.logger.summarize(self.model.global_step, summarizer="validation", scope="model", summaries_dict={
            "sonar_loss": np.mean(son_errs),
            "satellite_loss": np.mean(sat_errs)
        })

        self.logger.summarize(self.model.global_step, summarizer="validation", scope="metrics", summaries_dict={
            "accuracy": np.mean([negative_accuracy, positive_accuracy]),
            "positive_accuracy": positive_accuracy,
            "negative_accuracy": negative_accuracy
        })

        self.logger.summarize(self.model.global_step, summarizer="validation", scope="distribution", summaries_dict={
            "positive_mean": np.mean(positive_distances),
            "positive_std": np.std(positive_distances),
            "negative_mean": np.mean(negative_distances),
            "negative_std": np.std(negative_distances)
        })

        # Save and update epochs on the auxiliary model.
        self.sat_model.save_checkpoint()
        self.sat_model.epoch.assign_add(delta=1)

        """"
        CODE FOR: MULTIVARIATE GAUSSIAN MATCHING PROBABILITY

        # Calculate matching probabilities.
        positive_probabilities = []
        negative_probabilities = []
        norm = np.linalg.det(self.positives["std"]) / np.linalg.det(self.negatives["std"])
        pinv = np.linalg.pinv(self.positives["std"])
        ninv = np.linalg.pinv(self.negatives["std"])
        assert(positive_distances.shape[0] == negative_distances.shape[0])
        for i in range(positive_distances.shape[0]):
            positive_probabilities.append(1 / (1 + norm * np.exp(0.5 * np.dot(positive_distances[i] - self.positives["mean"], np.dot(pinv, np.transpose(positive_distances[i] - self.positives["mean"]))) - 0.5 * np.dot(positive_distances[i] - self.negatives["mean"], np.dot(ninv, np.transpose(positive_distances[i] - self.negatives["mean"]))))))
            negative_probabilities.append(1 / (1 + norm * np.exp(0.5 * np.dot(negative_distances[i] - self.positives["mean"], np.dot(pinv, np.transpose(negative_distances[i] - self.positives["mean"]))) - 0.5 * np.dot(negative_distances[i] - self.negatives["mean"], np.dot(ninv, np.transpose(negative_distances[i] - self.negatives["mean"]))))))

        # Debugging probabilities shapes
        positive_probabilities = np.array(positive_probabilities)
        negative_probabilities = np.array(negative_probabilities)

        self.logger.summarize(self.model.global_step, summarizer="validation", scope="distribution", summaries_dict={
            "positive_mean": np.linalg.norm(np.mean(positive_distances, axis=0)),
            "positive_std": np.linalg.norm(np.cov(np.transpose(positive_distances))),
            "negative_mean": np.linalg.norm(np.mean(negative_distances, axis=0)),
            "negative_std": np.linalg.norm(np.cov(np.transpose(negative_distances)))
        })
        """

    @tf.function
    def sonar_validate_step(self, data: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, ...]:
        x, y, _, z = data

        # Evaluate results on validation data.
        anc_pred = self.son_model(x, training=False)
        pos_pred = self.sat_model(y, training=False)
        neg_pred = self.sat_model(z, training=False)

        # Calculate triplet loss.
        loss = self.son_model.loss(anc_pred, (pos_pred, neg_pred))
        return loss, anc_pred, tf.linalg.norm(anc_pred - pos_pred, axis=1), tf.linalg.norm(anc_pred - neg_pred, axis=1)

    @tf.function
    def satellite_validate_step(self, data: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, ...]:
        y, x, z, _ = data

        # Evaluate results on validation data.
        anc_pred = self.sat_model(x, training=False)
        pos_pred = self.son_model(y, training=False)
        neg_pred = self.son_model(z, training=False)

        # Calculate triplet loss.
        loss = self.sat_model.loss(anc_pred, (pos_pred, neg_pred))
        return loss, anc_pred, pos_pred, neg_pred
