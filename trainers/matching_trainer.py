# matching_trainer.py

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
import os
import random as rand
import tensorflow as tf


class MatchingTrainer(BaseTrainer):
    def __init__(self, model: BaseModel, logger: BaseLogger, train_dataset: BaseDataset,
                 valid_dataset: Optional[BaseDataset]) -> None:
        super(MatchingTrainer, self).__init__(model, logger, train_dataset, valid_dataset)

    def train_epoch(self) -> None:
        loop = tqdm(range(len(self.train_dataset)))
        loop.set_description("Training Epoch [{}/{}]".format(int(self.model.epoch),
                                                             self.config.num_epochs))

        accs = []
        errs = []
        for data, _ in zip(self.train_dataset.data, loop):
            err, grad, acc = self.train_step(data)
            self.model.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

            # Append step data to epoch data list.
            accs.append(acc)
            errs.append(err)

            # Increment global step counter.
            self.model.global_step.assign_add(delta=1)

        self.logger.summarize(self.model.global_step, summarizer="train", summaries_dict={
            "accuracy": np.mean(accs),
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
            acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.greater_equal(z, 0.5), tf.greater_equal(prediction, 0.5)), dtype=tf.float32))

        grad = tape.gradient(loss, self.model.trainable_variables)
        return loss, grad, acc

    def validate_epoch(self) -> None:
        loop = tqdm(range(len(self.valid_dataset)))
        loop.set_description("Validating Epoch {}".format(int(self.model.epoch)))

        accs = []
        preds = []
        errs = []
        targets = []
        sats = []
        sons = []
        for data, _ in zip(self.valid_dataset.data, loop):
            err, pred, acc, target, sat, son = self.validate_step(data)

            # Append step data to epoch data list.
            accs.append(acc)
            errs.append(err)
            preds.append(pred)
            targets.append(target)
            sats.append(sat)
            sons.append(son)

        # Save an example batch.
        batch = rand.choice(range(len(preds)))
        path = os.path.join(self.config.evaluate_dir, "validation", str(int(self.model.epoch)))
        os.makedirs(path)
        for i in range(len(preds[batch])):
            # Save metrics.
            file = open(os.path.join(path, "%d.txt" % i), 'w')
            file.write("Prediction: %.5f\n" % float(preds[batch][i].numpy()))
            file.write("Target: %.5f\n" % float(targets[batch][i].numpy()))
            file.close()

            # Save sonar image.
            son = tf.image.encode_png(tf.cast(sons[batch][i].numpy() * 255.0, dtype=tf.uint8))
            tf.io.write_file(os.path.join(path, "son_%d.png" % i), son)

            # Save satellite image.
            sat = tf.image.encode_png(tf.cast(sats[batch][i].numpy() * 255.0, dtype=tf.uint8))
            tf.io.write_file(os.path.join(path, "sat_%d.png" % i), sat)

        self.logger.summarize(self.model.global_step, summarizer="validation", summaries_dict={
            "accuracy": np.mean(accs),
            "total_loss": np.mean(errs)
        })

    @tf.function
    def validate_step(self, data: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, ...]:
        x, y, z = data
        z = tf.expand_dims(z, axis=-1)

        # Evaluate results on validation data.
        prediction = self.model([x, y], training=False)
        loss = self.model.loss(z, prediction)
        acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.greater_equal(z, 0.5), tf.greater_equal(prediction, 0.5)), dtype=tf.float32))

        return loss, prediction, acc, z, y, x
