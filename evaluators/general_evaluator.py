# general_evaluator.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from base.base_dataset import BaseDataset
from base.base_evaluator import BaseEvaluator
from base.base_model import BaseModel
from tqdm import tqdm
from typing import Tuple

import os
import tensorflow as tf


class GeneralEvaluator(BaseEvaluator):
    def __init__(self, model: BaseModel, dataset: BaseDataset):
        super(GeneralEvaluator, self).__init__(model, dataset)

    def evaluate(self) -> None:
        loop = tqdm(range(len(self.dataset)))
        loop.set_description("Evaluating Examples")

        i = 0
        for data, _ in zip(self.dataset.data, loop):
            pred, _, _, _ = self.evaluate_step(data)

            # Save image.
            raw = tf.image.encode_png(pred, compression=0)
            tf.io.write_file(os.path.join(self.config.evaluate_dir, "{:05d}.png".format(i)), raw)

            i += 1

    @tf.function
    def evaluate_step(self, data: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, ...]:
        x, y, z = data

        # Evaluate results on testing data.
        prediction = self.model([x, y], training=False)

        return prediction, z, y, x
