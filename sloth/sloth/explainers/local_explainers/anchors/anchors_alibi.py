import logging
import warnings
import alibi
import numpy as np
import pandas as pd
from typing import Union, List
from sloth.features import DataType
from sloth.explainers.local_explainers.anchors.anchors import Anchors
from sloth.validation_task import ValidationTask


class AnchorsAlibi(Anchors):
    def __init__(self,
                 task: ValidationTask,
                 threshold= 0.95,
                 epsilon_output= 0.01,
                 percentiles= None,
                 delta= 0.01,
                 tau= 0.6,
                 batch_size= 100,
                 coverage_samples= 10000,
                 beam_size= 3,
                 stop_on_first= False,
                 max_anchor_size= None,
                 min_samples_start= 100,
                 n_covered_ex= 10,
                 binary_cache_size= 10000,
                 cache_margin= 1000,
                 verbose= False,
                 verbose_every= 1,
                 seed= 42,
                 **kwargs
                 ):

        super().__init__(task=task,
                         threshold=threshold,
                         epsilon_output=epsilon_output,
                         percentiles=percentiles,
                         delta=delta,
                         tau=tau,
                         batch_size=batch_size,
                         coverage_samples=coverage_samples,
                         beam_size=beam_size,
                         stop_on_first=stop_on_first,
                         max_anchor_size=max_anchor_size,
                         min_samples_start=min_samples_start,
                         n_covered_ex=n_covered_ex,
                         binary_cache_size=binary_cache_size,
                         cache_margin=cache_margin,
                         verbose=verbose,
                         verbose_every=verbose_every,
                         seed=seed,
                         **kwargs)

    def _explain(self, p, predict):
        explainer = alibi.explainers.AnchorTabular(predict,
                                                   self.feature_names,
                                                   categorical_names=self.categorical_names,
                                                   seed=self.seed,
                                                   ohe=self.ohe)
        explainer.fit(self.task.data, self.percentiles)  # TODO move this to constructor? after dealing with regression case
        explanation = explainer.explain(p,
                                        delta=self.delta,
                                        tau=self.tau,
                                        batch_size=self.batch_size,
                                        coverage_samples=self.coverage_samples,
                                        beam_size=self.beam_size,
                                        stop_on_first=self.stop_on_first,
                                        max_anchor_size=self.max_anchor_size,
                                        min_samples_start=self.min_samples_start,
                                        n_covered_ex=self.n_covered_ex,
                                        binary_cache_size=self.binary_cache_size,
                                        cache_margin=self.cache_margin,
                                        verbose=self.verbose,
                                        verbose_every=self.verbose_every)
        return {'anchor': ' AND '.join(explanation.anchor),
                'precision': explanation.precision,
                'coverage': explanation.coverage}


if __name__ == '__main__':
    import sloth

    # get a sample validation task from a synthetic credit default model
    # validation_task = sloth.datasets.test_sets.simple_classification_ordinal_categorical(n_samples=10_000)
    validation_task = sloth.datasets.test_sets.simple_classification_ordinal(n_samples=1_000, x=0, f=0)
    anchors = sloth.AnchorsAlibi(validation_task)
    data = validation_task.data
    x = data[0]
    # df = anchors.explain(x=x) #TODO how to explain?
    # print(df)
