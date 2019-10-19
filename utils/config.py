# config.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging
from typing import Tuple

import namegenerator
import os
import time


# Core entries
flags.DEFINE_enum("mode", "train", ["evaluate", "restore", "train"], "The modes that are available.")
flags.DEFINE_string("name", "auto", "Name of the folder to store the files of the running experiment.")

# Network entries
flags.DEFINE_float("learning_rate", 2e-4, "Initial learning rate for the chosen optimizer")
flags.DEFINE_integer("batch_size", 8, "The size of the batch to use while training the network.", lower_bound=1)
flags.DEFINE_integer("filters", 8, "A parameter that scales the depth of the neural network.", lower_bound=1)
flags.DEFINE_integer("num_epochs", 100, "Number of epochs to train the network for.", lower_bound=1)

# Data entries
flags.DEFINE_list("input_shape", [256, 128, 3], "The shape of the data to input in the neural network.")
flags.DEFINE_list("satellite_shape", [512, 256, 3], "The shape of the satellite image to input in the network.")
flags.DEFINE_list("output_shape", [256, 128, 3], "The shape of the data that will be output from the neural network.")

# Non-configurable entries
flags.DEFINE_string("checkpoint_dir", "", "Location to save the training checkpoints. (Do not edit).")
flags.DEFINE_string("evaluate_dir", "", "Location to save the evaluation results. (Do not edit).")
flags.DEFINE_string("execution_dir", "", "Location to save the execution information. (Do not edit).")
flags.DEFINE_string("presentation_dir", "", "Location to save the results in presentable format. (Do not edit).")
flags.DEFINE_string("summary_dir", "", "Location to save the training summaries. (Do not edit).")


def create_directories(config: flags.FlagValues, directories: Tuple[str, ...]) -> None:
    for d in directories:
        if not os.path.exists(d):
            os.makedirs(d)
        else:
            if config.mode == "train":
                logging.fatal("Cannot train with chosen name because directory already exists.")


def process_config() -> flags.FlagValues:
    config = flags.FLAGS

    # Process running experiment name.
    if (config.mode == "evaluate" or config.mode == "restore") and config.name == "auto":
        logging.fatal("Cannot automatically generate a name for the chosen mode.")
    elif config.mode == "train" and config.name == "auto":
        config.name = os.path.join(time.strftime("%Y-%m-%d"), namegenerator.gen())
        logging.info("Experiment Name: {}".format(config.name))

    # Convert shapes to tuples.
    config.input_shape = tuple(config.input_shape)
    config.satellite_shape = tuple(config.satellite_shape)
    config.output_shape = tuple(config.output_shape)

    # Set directories to their appropriate paths.
    config.execution_dir = os.path.join(os.curdir, "executions", config.name)
    config.checkpoint_dir = os.path.join(config.execution_dir, "checkpoint")
    config.evaluate_dir = os.path.join(config.execution_dir, "result")
    config.log_dir = os.path.join(config.execution_dir, "log")
    config.presentation_dir = os.path.join(config.execution_dir, "presentation")
    config.summary_dir = os.path.join(config.execution_dir, "summary")

    create_directories(config, directories=(config.checkpoint_dir, config.evaluate_dir, config.log_dir,
                                            config.presentation_dir, config.summary_dir))

    # Log out the command for using TensorBoard.
    logging.info('tensorboard --logdir="{}" --bind_all --port 6006'.format(os.path.abspath(config.summary_dir)))
    return config
