# triplet_dataset.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from base.base_dataset import BaseDataset
from typing import Tuple

import glob
import os
import pandas as pd
import tensorflow as tf


class TripletDataset(BaseDataset):
    def __init__(self, batch_size: int, folder: str, x_shape: Tuple[int, int, int], y_shape: Tuple[int, int, int],
                 is_evaluating=False):
        self.x_shape = x_shape
        self.y_shape = y_shape

        # Acquire image filenames.
        if is_evaluating:
            self._x_filenames = sorted(glob.glob(
                os.path.join(os.curdir, folder, "..", "images", "sonar.evaluate", "*.png")))
            self._y_filenames = sorted(glob.glob(
                os.path.join(os.curdir, folder, "..", "images", "satellite.evaluate", "*.png")))
        else:
            self._x_filenames = sorted(glob.glob(
                os.path.join(os.curdir, folder, "..", "images", "sonar", "*.png")))
            self._y_filenames = sorted(glob.glob(
                os.path.join(os.curdir, folder, "..", "images", "satellite", "*.png")))

        self._sonar_data = pd.read_csv(os.path.join(os.curdir, folder, "sonar", "data.csv")).iloc[:, :].values
        self._satellite_data = pd.read_csv(os.path.join(os.curdir, folder, "satellite", "data.csv")).iloc[:, :].values

        assert(len(self._x_filenames) == len(self._y_filenames))
        assert(len(self._sonar_data) == len(self._satellite_data))
        super(TripletDataset, self).__init__(batch_size=batch_size, length=len(self._sonar_data[:, 0]),
                                             types=(tf.string, tf.string, tf.string,
                                                    tf.string, tf.string, tf.string))

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
        return self._load_x(args[0]), self._load_y(args[1]), self._load_y(args[2]), \
               self._load_y(args[3]), self._load_x(args[4]), self._load_x(args[5])

    def next_entry(self) -> Tuple[str, ...]:
        for i in range(len(self._sonar_data)):
            yield (self._x_filenames[int(self._sonar_data[i, 0])],
                   self._y_filenames[int(self._sonar_data[i, 1])],
                   self._y_filenames[int(self._sonar_data[i, 2])],
                   self._y_filenames[int(self._satellite_data[i, 0])],
                   self._x_filenames[int(self._satellite_data[i, 1])],
                   self._x_filenames[int(self._satellite_data[i, 2])])
