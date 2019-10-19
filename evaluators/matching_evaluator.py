# matching_evaluator.py

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


class MatchingEvaluator(BaseEvaluator):
    def __init__(self, model: BaseModel, dataset: BaseDataset):
        super(MatchingEvaluator, self).__init__(model, dataset)

        # Evaluator state variables.
        self.example = 0

    def evaluate(self) -> None:
        loop = tqdm(range(len(self.dataset)))
        loop.set_description("Evaluating Example [{}/{}]".format(self.example, len(self.dataset)))

        for data, _ in zip(self.dataset.data, loop):
            pred, _, _, _ = self.evaluate_step(data)

            # Save metrics.
            file = open(os.path.join(self.config.evaluate_dir, "matching_{:05d}.txt".format(self.example)), 'w')
            file.write("Prediction: %.5f\n" % float(pred.numpy()))
            file.close()

            # Increment example counter.
            self.example += 1
            loop.set_description("Evaluating Example [{}/{}]".format(self.example, len(self.dataset)))

    @tf.function
    def evaluate_step(self, data: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, ...]:
        x, y, z = data
        z = tf.expand_dims(z, axis=-1)

        # Evaluate results on testing data.
        prediction = self.model([x, y], training=False)

        return prediction, z, y, x
