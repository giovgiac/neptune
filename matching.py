# matching.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import logging

from datasets.matching_dataset import MatchingDataset
from evaluators.matching_evaluator import MatchingEvaluator
from loggers.logger import Logger
from models.dizygotic_net import DizygoticNet
from optimizers.lookahead import Lookahead
from optimizers.radam import RectifiedAdam
from trainers.matching_trainer import MatchingTrainer
from utils.config import process_config

import tensorflow as tf


def main(argv) -> None:
    del argv

    # Process the configuration from flags.
    config = process_config()

    if config.mode != "evaluate":
        # Define the datasets.
        train_dataset = MatchingDataset(batch_size=config.batch_size,
                                        folder="datasets/matching_aracati/train",
                                        x_shape=config.input_shape,
                                        y_shape=config.output_shape,
                                        is_evaluating=False)

        valid_dataset = MatchingDataset(batch_size=config.batch_size,
                                        folder="datasets/matching_aracati/validation",
                                        x_shape=config.input_shape,
                                        y_shape=config.output_shape,
                                        is_evaluating=False)

        # Define the model.
        loss = tf.keras.losses.BinaryCrossentropy()
        ranger = Lookahead(RectifiedAdam(learning_rate=config.learning_rate), sync_period=6, slow_step_size=0.5)
        model = DizygoticNet(filters=config.filters, loss=loss, optimizer=ranger)
        if config.mode == "restore":
            model.load_checkpoint()

        # Define the logger.
        logger = Logger()

        # Define the trainer.
        trainer = MatchingTrainer(model=model, logger=logger, train_dataset=train_dataset, valid_dataset=valid_dataset)
        trainer.train()
    else:
        # Define the test dataset.
        test_dataset = MatchingDataset(batch_size=1,
                                       folder="datasets/matching_aracati/test",
                                       x_shape=config.input_shape,
                                       y_shape=config.output_shape,
                                       is_evaluating=True)

        # Define the model.
        model = DizygoticNet(filters=config.filters, loss=None, optimizer=None)
        model.load_checkpoint()

        # Define the evaluator.
        evaluator = MatchingEvaluator(model=model, dataset=test_dataset)
        evaluator.evaluate()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
