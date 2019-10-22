# triplet.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

from datasets.triplet_dataset import TripletDataset
from loggers.logger import Logger
from losses.triplet import TripletLoss
from models.encode_net import EncodeNet
from optimizers.lookahead import Lookahead
from optimizers.radam import RectifiedAdam
from trainers.triplet_trainer import TripletTrainer
from utils.config import process_config


# Network entries
flags.DEFINE_float("learning_rate", 2e-4, "Initial learning rate for the chosen optimizer")
flags.DEFINE_integer("batch_size", 32, "The size of the batch to use while training the network.", lower_bound=1)
flags.DEFINE_integer("filters", 4, "A parameter that scales the depth of the neural network.", lower_bound=1)
flags.DEFINE_integer("num_epochs", 100, "Number of epochs to train the network for.", lower_bound=1)

# Data entries
flags.DEFINE_list("input_shape", [128, 256, 1], "The shape of the data to input in the neural network.")
flags.DEFINE_list("satellite_shape", [128, 256, 1], "The shape of the satellite image to input in the network.")
flags.DEFINE_list("output_shape", [128, 256, 1], "The shape of the data that will be output from the neural network.")


def main(argv) -> None:
    del argv

    # Process the configuration from flags.
    config = process_config()

    if config.mode != "evaluate":
        # Define the datasets.
        train_dataset = TripletDataset(batch_size=config.batch_size,
                                       folder="datasets/triplet_aracati/train",
                                       x_shape=config.input_shape,
                                       y_shape=config.output_shape,
                                       is_evaluating=False)

        valid_dataset = TripletDataset(batch_size=config.batch_size,
                                       folder="datasets/triplet_aracati/validation",
                                       x_shape=config.input_shape,
                                       y_shape=config.output_shape,
                                       is_evaluating=False)

        # Define the sonar model.
        son_loss = TripletLoss()
        son_ranger = Lookahead(RectifiedAdam(learning_rate=config.learning_rate), sync_period=6, slow_step_size=0.5)
        son_model = EncodeNet(filters=config.filters, loss=son_loss, optimizer=son_ranger)
        if config.mode == "restore":
            son_model.load_checkpoint()

        # Define the satellite model.
        sat_loss = TripletLoss()
        sat_ranger = Lookahead(RectifiedAdam(learning_rate=config.learning_rate), sync_period=6, slow_step_size=0.5)
        sat_model = EncodeNet(filters=config.filters, loss=sat_loss, optimizer=sat_ranger)
        if config.mode == "restore":
            sat_model.load_checkpoint()

        # Define the logger.
        logger = Logger()

        # Define the trainer.
        trainer = TripletTrainer(son_model=son_model, sat_model=sat_model, logger=logger, train_dataset=train_dataset,
                                 valid_dataset=valid_dataset)
        trainer.train()
    else:
        logging.fatal("Evaluation not implemented yet.")


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
