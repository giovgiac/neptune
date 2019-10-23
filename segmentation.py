# segmentation.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

from datasets.segmentation_dataset import SegmentationDataset
from loggers.logger import Logger
from models.unet import UNet
from optimizers.lookahead import Lookahead
from optimizers.radam import RectifiedAdam
from trainers.segmentation_trainer import SegmentationTrainer
from utils.config import process_config

import tensorflow as tf


# Network entries
flags.DEFINE_float("learning_rate", 2e-4, "Initial learning rate for the chosen optimizer")
flags.DEFINE_integer("batch_size", 8, "The size of the batch to use while training the network.", lower_bound=1)
flags.DEFINE_integer("filters", 16, "A parameter that scales the depth of the neural network.", lower_bound=1)
flags.DEFINE_integer("num_epochs", 100, "Number of epochs to train the network for.", lower_bound=1)

# Data entries
flags.DEFINE_list("input_shape", [256, 256, 3], "The shape of the data to input in the neural network.")
flags.DEFINE_list("satellite_shape", [256, 256, 3], "The shape of the satellite image to input in the network.")
flags.DEFINE_list("output_shape", [256, 256, 3], "The shape of the data that will be output from the neural network.")


def main(argv) -> None:
    del argv

    # Process the configuration from flags.
    config = process_config()

    if config.mode != "evaluate":
        # Define the datasets.
        train_dataset = SegmentationDataset(batch_size=config.batch_size,
                                            folder="datasets/segmentation_aracati/train",
                                            x_shape=config.input_shape,
                                            y_shape=config.output_shape)

        valid_dataset = SegmentationDataset(batch_size=config.batch_size,
                                            folder="datasets/segmentation_aracati/validation",
                                            x_shape=config.input_shape,
                                            y_shape=config.output_shape)

        # Define the model.
        loss = tf.keras.losses.CategoricalCrossentropy()
        ranger = Lookahead(RectifiedAdam(learning_rate=config.learning_rate), sync_period=6, slow_step_size=0.5)
        model = UNet(filters=config.filters, loss=loss, optimizer=ranger)
        if config.mode == "restore":
            model.load_checkpoint()

        # Define the logger.
        logger = Logger()

        # Define the trainer.
        trainer = SegmentationTrainer(model=model, logger=logger, train_dataset=train_dataset,
                                      valid_dataset=valid_dataset)
        trainer.train()
    else:
        logging.fatal("Evaluation not implemented yet.")


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
