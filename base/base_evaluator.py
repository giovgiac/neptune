# base_evaluator.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from base.base_dataset import BaseDataset
from base.base_model import BaseModel

import abc


class BaseEvaluator(abc.ABC):
    def __init__(self, model: BaseModel, dataset: BaseDataset) -> None:
        self.config = flags.FLAGS
        self.model = model
        self.dataset = dataset

    @abc.abstractmethod
    def evaluate(self) -> None:
        raise NotImplementedError
