# triplet_evaluator.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from base.base_dataset import BaseDataset
from base.base_evaluator import BaseEvaluator
from base.base_model import BaseModel
from tqdm import tqdm
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf


class TripletEvaluator(BaseEvaluator):
    def __init__(self, son_model: BaseModel, sat_model: BaseModel, dataset: BaseDataset, positives: Dict, negatives: Dict):
        super(TripletEvaluator, self).__init__({"son_model": son_model, "sat_model": sat_model}, dataset)

        # Neural network model references.
        self.son_model = son_model
        self.sat_model = sat_model

        # Store parameters.
        self.positives = positives
        self.negatives = negatives

        # Evaluator state variables.
        self.example = 0

    def evaluate(self) -> None:
        loop = tqdm(range(len(self.dataset)))
        loop.set_description("Evaluating Example [{}/{}]".format(self.example, len(self.dataset)))

        errs = []
        positive_distances = []
        negative_distances = []
        for data, _ in zip(self.dataset.data, loop):
            err, anc, pos, neg = self.evaluate_step(data)

            # Append step data to list.
            errs.append(err)
            positive_distances.append(np.linalg.norm(anc - pos))
            negative_distances.append(np.linalg.norm(anc - neg))

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
        accuracy = np.mean([negative_accuracy, positive_accuracy])

        print("Triplet Loss: {:.2f}".format(np.mean(errs)))
        print("Overall Accuracy: {:.2f}".format(accuracy))
        print("Positive Accuracy: {:.2f}".format(positive_accuracy))
        print("Negative Accuracy: {:.2f}".format(negative_accuracy))

    @tf.function
    def evaluate_step(self, data: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, ...]:
        x, y, _, z = data

        # Evaluate results on testing data.
        anc_pred = self.son_model(x, training=False)
        pos_pred = self.sat_model(y, training=False)
        neg_pred = self.sat_model(z, training=False)

        # Calculate triplet loss.
        loss = self.son_model.loss(anc_pred, (pos_pred, neg_pred))
        return loss, anc_pred, pos_pred, neg_pred
