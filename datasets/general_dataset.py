# general_dataset.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from base.base_dataset import BaseDataset
from typing import List, Tuple

import glob
import os
import tensorflow as tf


class GeneralDataset(BaseDataset):
    def __init__(self, batch_size: int, folder: str, x_shape: Tuple[int, int, int], y_shape: Tuple[int, int, int],
                 z_shape: Tuple[int, int, int]):
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.z_shape = z_shape

        # Acquire image filenames.
        self._x_filenames = sorted(glob.glob(os.path.join(os.curdir, folder, "input", "*", "*.png")))
        self._y_filenames = self._extract_satellites(folder, self._x_filenames)
        self._z_filenames = sorted(glob.glob(os.path.join(os.curdir, folder, "gt", "*", "*.png")))

        assert (len(self._x_filenames) == len(self._y_filenames) == len(self._z_filenames))
        super(GeneralDataset, self).__init__(batch_size=batch_size, length=len(self._x_filenames),
                                             types=(tf.string, tf.string, tf.string))

    @staticmethod
    def _extract_satellites(folder: str, filenames: List[str]):
        satellites = []
        for name in filenames:
            satellites.append(str(os.path.join(os.curdir, folder, "satellite", name.split(os.sep)[-2])) + ".png")

        return satellites

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
    def _load_z(self, filename: str) -> tf.Tensor:
        raw = tf.io.read_file(filename)
        img = tf.image.decode_image(raw, channels=self.z_shape[2], dtype=tf.float32, expand_animations=False)

        return img

    @tf.function
    def load_entry(self, *args) -> Tuple[tf.Tensor, ...]:
        return self._load_x(args[0]), self._load_y(args[1]), self._load_z(args[2])

    def next_entry(self) -> Tuple[str, ...]:
        for i in range(len(self._x_filenames)):
            yield (self._x_filenames[i], self._y_filenames[i], self._z_filenames[i])
