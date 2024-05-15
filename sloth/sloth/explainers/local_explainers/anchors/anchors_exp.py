import logging
import warnings
import numpy as np
import pandas as pd
from typing import Union, List

from anchor import anchor_tabular
from sloth.features import DataType
from sloth.explainers.local_explainers.anchors.anchors import Anchors
from sloth.validation_task import ValidationTask


class AnchorsExp(Anchors):

    def __init__(self, 
                 task: ValidationTask,
                 threshold=0.95,
                 epsilon_output=0.01,
                 discretizer='quartile',
                 seed=42,
                 **kwargs
                 ):
        """Anchor computation using the :py:mod:`anchor-exp` package that can be found on GitHub, see `here <https://github.com/marcotcr/anchor>`_.

        Args:
            task (ValidationTask): The validation task.
            threshold (float, optional): The accuracy that must be obtained by the anchor. Defaults to 0.95 (which means that 95% of the data points must be correctly classified by the anchor).
            epsilon_output (float, optional): Used for regression problems. Since anchors can only be computed for classification problems, that threshold is used to transform the regression to a classification problem, i.e. all points whose predicted values  are within the distance to the value at the point of interest are classified by True. Defaults to 0.01.
            discretizer (str, optional): Must be either 'quartile' or 'decile' and determines the grids for the discretized variables. Defaults to 'quartile'.
            seed (int, optional): Defaults to 4.

        See Also
        --------
            AnchorsAlibi : Anchor computation using the algorithm from the module :py:mod:`alibi`.

        Examples:
            >>> import sloth
            >>> validation_task = sloth.datasets.test_sets.simple_classification_ordinal(n_samples=10_000)
            >>> anchors = sloth.explainers.local_explainers.anchors.AnchorsExp(validation_task, discretizer='decile')
            >>> print(anchors.explain(x=validation_task.data[11]))
                    anchor                      precision  coverage           point                                 target
                0  x_2 > 0.61 AND x_1 > 0.79        1.0    0.0187  [0.8977710745066665, 0.9312640661491187, 0.616...  [True]
       
        """

        super().__init__(task,
                         threshold=threshold,
                         epsilon_output=epsilon_output,
                         discretizer=discretizer,
                         seed=seed,
                         **kwargs
                         )

    def _explain(self, p, predict):
        explainer = anchor_tabular.AnchorTabularExplainer(['0', '1'],
                                                          self.feature_names,
                                                          self.task.data,
                                                          self.categorical_names, discretizer=self.discretizer)
        explanation = explainer.explain_instance(p,
                                                 predict,
                                                 threshold=self.threshold)
        return {'anchor': ' AND '.join(explanation.names()),
                'precision': explanation.precision(),
                'coverage': explanation.coverage()}


if __name__ == '__main__':
    import sloth

    validation_task = sloth.datasets.test_sets.simple_classification_ordinal(n_samples=10_000)
    anchors = sloth.AnchorsExp(validation_task)
    # print(anchors.explain(x=validation_task.data[11])) #TODO how to explain
