# segmentation_dataset.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from base.base_dataset import BaseDataset
from typing import Tuple

import glob
import os
import tensorflow as tf


class SegmentationDataset(BaseDataset):
    def __init__(self, batch_size: int, folder: str, x_shape: Tuple[int, int, int], y_shape: Tuple[int, int, int]):
        self.x_shape = x_shape
        self.y_shape = y_shape

        # Acquire image filenames.
        self._x_filenames = sorted(glob.glob(os.path.join(os.curdir, folder, "input", "*.png")))
        self._y_filenames = sorted(glob.glob(os.path.join(os.curdir, folder, "gt", "*.png")))

        assert (len(self._x_filenames) == len(self._y_filenames))
        super(SegmentationDataset, self).__init__(batch_size=batch_size, length=len(self._x_filenames),
                                                  types=(tf.string, tf.string))

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
        return self._load_x(args[0]), self._load_y(args[1])

    def next_entry(self) -> Tuple[str, ...]:
        for i in range(len(self._x_filenames)):
            yield (self._x_filenames[i], self._y_filenames[i])
