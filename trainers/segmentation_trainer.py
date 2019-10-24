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
        metrics = {"psnr": [], "ssim": []}
        for data, _ in zip(self.valid_dataset.data, loop):
            err, prediction, target, psnr, ssim = self.validate_step(data)

            # Convert prediction and target to desired format.
            prediction = tf.argmax(prediction, axis=-1)
            target = tf.argmax(target, axis=-1)

            # Append step data to epoch data list.
            errs.append(err)
            predictions.append(prediction)
            targets.append(target)
            metrics["psnr"].append(psnr)
            metrics["ssim"].append(ssim)

        # Output validation loss and images to TensorBoard.
        batch = rand.choice(range(len(predictions)))
        self.logger.summarize(self.model.global_step, summarizer="validation", summaries_dict={
            "prediction": tf.keras.utils.to_categorical(predictions[batch], num_classes=3),
            "target": tf.keras.utils.to_categorical(targets[batch], num_classes=3),
        })

        # Convert lists to proper arrays.
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        # Calculate metrics
        accuracy = tf.reduce_mean(tf.cast(tf.equal(targets, predictions), dtype=tf.float32))
        iou_num = np.sum(np.logical_and(np.ravel(tf.keras.utils.to_categorical(targets, num_classes=3)),
                         np.ravel(tf.keras.utils.to_categorical(predictions, num_classes=3))))
        iou_den = np.sum(np.logical_or(np.ravel(tf.keras.utils.to_categorical(targets, num_classes=3)),
                         np.ravel(tf.keras.utils.to_categorical(predictions, num_classes=3))))
        precision, recall, fscore, _ = precision_recall_fscore_support(targets.ravel(), predictions.ravel())

        # Categorize validation metrics under TensorBoard.
        self.logger.summarize(self.model.global_step, summarizer="validation", scope="metrics", summaries_dict={
            "accuracy": accuracy,
            "iou": iou_num / iou_den,
            "psnr": np.mean(metrics["psnr"]),
            "ssim": np.mean(metrics["ssim"])
        })

        self.logger.summarize(self.model.global_step, summarizer="validation", scope="model", summaries_dict={
            "total_loss": np.mean(errs)
        })

        for i, scope in zip(range(3), ["movable", "stationary", "water"]):
            self.logger.summarize(self.model.global_step, summarizer="validation", scope=scope, summaries_dict={
                "precision": precision[i],
                "recall": recall[i],
                "f-score": fscore[i]
            })

    @tf.function
    def validate_step(self, data: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, ...]:
        x, y = data

        # Evaluate results on validation data.
        prediction = self.model(x, training=False)
        loss = self.model.loss(y, prediction)

        # Calculate metrics on validation data.
        psnr = tf.image.psnr(y, prediction, max_val=1.0)
        ssim = tf.image.ssim(y, prediction, max_val=1.0)

        return loss, prediction, y, psnr, ssim
