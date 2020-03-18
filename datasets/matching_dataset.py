# matching_dataset.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from base.base_dataset import BaseDataset
from typing import Tuple

import glob
import os
import pandas as pd
import tensorflow as tf


class MatchingDataset(BaseDataset):
    def __init__(self, batch_size: int, folder: str, x_shape: Tuple[int, int, int], y_shape: Tuple[int, int, int],
                 is_evaluating=False):
        self.x_shape = x_shape
        self.y_shape = y_shape

        # Acquire image filenames.
        if is_evaluating:
            self._x_filenames = sorted(glob.glob(os.path.join(os.curdir, folder, "..", "images", "input.evaluate", "*.png")))
            self._y_filenames = sorted(glob.glob(os.path.join(os.curdir, folder, "..", "images", "gt.evaluate", "*.png")))
        else:
            self._x_filenames = sorted(glob.glob(os.path.join(os.curdir, folder, "..", "images", "input", "*.png")))
            self._y_filenames = sorted(glob.glob(os.path.join(os.curdir, folder, "..", "images", "gt", "*.png")))
        self._data = pd.read_csv(os.path.join(os.curdir, folder, "data.csv")).iloc[:, :].values

        assert(len(self._x_filenames) == len(self._y_filenames))
        super(MatchingDataset, self).__init__(batch_size=batch_size, length=len(self._data[:, 0]),
                                              types=(tf.string, tf.string, tf.float32))

    @tf.function
    def _load_x(self, filename: str) -> tf.Tensor:
        raw = tf.io.read_file(filename)
        img = tf.image.decode_image(raw, channels=self.x_shape[2], dtype=tf.float32, expand_animations=False)

        return img

    @tf.function
    def _load_y(self, filename: str) -> tf.Tensor:
        raw = tf.io.read_file(filename)
        img = tf.image.decode_image(raw, channels=self.y_shape[2], dtype=tf.float32, expand_animations=False)

        return img

    @tf.function
    def load_entry(self, *args) -> Tuple[tf.Tensor, ...]:
        return self._load_x(args[0]), self._load_y(args[1]), args[2]

    def next_entry(self) -> Tuple[str, ...]:
        for i in range(len(self._data)):
            yield (self._x_filenames[int(self._data[i, 0])],
                   self._y_filenames[int(self._data[i, 1])],
                   self._data[i, 2])
