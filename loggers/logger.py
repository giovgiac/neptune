# logger.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict

from base.base_logger import BaseLogger

import os
import tensorflow as tf


class Logger(BaseLogger):
    def __init__(self) -> None:
        super(Logger, self).__init__()

        # Find path for saving summaries and accessing TensorBoard.
        train_path = os.path.join(self.config.summary_dir, "train")
        valid_path = os.path.join(self.config.summary_dir, "validation")

        # Create the summary writers.
        self.train_summary = tf.summary.create_file_writer(train_path)
        self.valid_summary = tf.summary.create_file_writer(valid_path)

        # Enable graph and logging for the model.
        tf.summary.trace_on(graph=True, profiler=False)

    def summarize(self, step: tf.Variable, summarizer="train", scope="", summaries_dict: Dict = None) -> None:
        summary = self.train_summary if summarizer == "train" else self.valid_summary
        with tf.name_scope(scope):
            if summaries_dict is not None:
                for tag, value in summaries_dict.items():
                    with summary.as_default():
                        if len(value.shape) <= 1:
                            tf.summary.scalar(tag, value, step=step)
                        else:
                            tf.summary.image(tag, value, step=step, max_outputs=8)
                summary.flush()
