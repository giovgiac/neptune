# main.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import logging

from datasets.general_dataset import GeneralDataset
from evaluators.general_evaluator import GeneralEvaluator
from loggers.logger import Logger
from models.wnet import WNet
from optimizers.radam import RAdam
from trainers.general_trainer import GeneralTrainer
from utils.config import process_config

import tensorflow as tf


def main(argv) -> None:
    del argv

    # Process the configuration from flags.
    config = process_config()

    if config.mode != "evaluate":
        # Define the datasets.
        train_dataset = GeneralDataset(batch_size=config.batch_size,
                                       folder="datasets/general_aracati/train",
                                       x_shape=config.input_shape,
                                       y_shape=config.satellite_shape,
                                       z_shape=config.output_shape)

        valid_dataset = GeneralDataset(batch_size=config.batch_size,
                                       folder="datasets/general_aracati/validation",
                                       x_shape=config.input_shape,
                                       y_shape=config.satellite_shape,
                                       z_shape=config.output_shape)

        # Define the model.
        loss = tf.keras.losses.MeanAbsoluteError()
        optimizer = RAdam(learning_rate=config.learning_rate)
        model = WNet(filters=config.filters, loss=loss, optimizer=optimizer)
        if config.mode == "restore":
            model.load_checkpoint()

        # Define the logger.
        logger = Logger()

        # Define the trainer.
        trainer = GeneralTrainer(model=model, logger=logger, train_dataset=train_dataset, valid_dataset=valid_dataset)
        trainer.train()
    else:
        # Define the test dataset.
        test_dataset = GeneralDataset(batch_size=1,
                                      folder="datasets/general_aracati/test",
                                      x_shape=config.input_shape,
                                      y_shape=config.satellite_shape,
                                      z_shape=config.output_shape)

        # Define the model.
        model = WNet(filters=config.filters, loss=None, optimizer=None)
        model.load_checkpoint()

        # Define the evaluator.
        evaluator = GeneralEvaluator(model=model, dataset=test_dataset)
        evaluator.evaluate()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
