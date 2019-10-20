# base_dataset.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from typing import Tuple

import abc
import multiprocessing
import tensorflow as tf


class BaseDataset(abc.ABC):
    def __init__(self, batch_size: int, length: int, types: Tuple[tf.DType, ...]) -> None:
        self.batch_size = batch_size
        self.config = flags.FLAGS
        self.length = length

        # Create and configure TensorFlow dataset.
        self.data = tf.data.Dataset.from_generator(generator=self.next_entry, output_types=types)
        self.data = self.data.shuffle(buffer_size=self.length, reshuffle_each_iteration=True)
        self.data = self.data.repeat()
        self.data = self.data.map(map_func=self.load_entry, num_parallel_calls=multiprocessing.cpu_count())
        self.data = self.data.batch(batch_size=batch_size, drop_remainder=True)
        self.data = self.data.prefetch(buffer_size=multiprocessing.cpu_count())

    def __len__(self) -> int:
        return self.length // self.batch_size

    @abc.abstractmethod
    @tf.function
    def load_entry(self, *args) -> Tuple[tf.Tensor, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def next_entry(self) -> Tuple[str, ...]:
        raise NotImplementedError
