from __future__ import annotations
import abc
import numpy as np
import logging
import sloth.result_cache as result_cache
from sloth.validation_task import ValidationTask
from sloth.explainers.base_explainer import BaseExplainer, BaseExplanation
from sloth.explainers.datapoints.examples import Examples



logger = logging.getLogger("sloth")


class BaseLocalExplainer(BaseExplainer):
    def __init__(self, task: ValidationTask, **kwargs):
        super().__init__(task, **kwargs)

    @abc.abstractmethod
    def _explain(self, x: np.ndarray) -> BaseExplanation:
        pass

    def explain(self, x: np.ndarray | Examples) -> BaseExplanation:
        if isinstance(x, Examples):
            x = self.task.data[x.points]
        if x is None:
            x = self.task.data[
                np.random.choice(
                    self.task.data.shape[0], size=self.max_samples, replace=False
                ),
                :,
            ]
        hashkey = (
            result_cache._create_hashkey_np(x) + self._hashkey
        )  # TODO x muss array sein, kein df möglich
        if result_cache.has_result(hashkey):
            logger.info("Using cached result for hashkey  %s", hashkey)
            return result_cache.get_result(hashkey)
        else:
            result = self._explain(x)
            result_cache.add_result(hashkey, result)
            return result
