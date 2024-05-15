from typing import List
import numpy as np
import logging
import json
import os

try:
    import hiplot as hip
    has_hiplot = True
except ImportError:
    has_hiplot = False

from sloth.features import DataType
from sloth.explainers.utils import UtilsClass

logger = logging.getLogger('sloth')

class DataPoints:
    def __init__(self, validation_task, **kwargs):
        self.task = validation_task
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _get_coordinates(self) -> List[int]:
        result = []
        for f in self.task.input_features.values():
            if f.data_type == DataType.ORDINAL.value:
                result.append(np.argmax(self.task.data[:, f.column]))
                result.append(np.argmin(self.task.data[:, f.column]))
        tmp = list(set(result))
        return tmp


class ExtremePoints(DataPoints):
    def __init__(self, task=None, **kwargs):
        super().__init__(validation_task=task,**kwargs)

    def _get_coordinate_extreme_points(self) -> List[int]:
        logger.info('Choosing datapoints with extreme input (min and max) values as examples.')
        tmp = self._get_coordinates()
        logger.debug(str(len(tmp)) + ' points with extreme input values chosen as parameter.')
        return tmp


class ExtremePredictions(DataPoints):

    def __init__(self,
                 task=None,
                 quantile=0.01,
                 max_n_points=5,
                 **kwargs):
        super().__init__(validation_task=task,
                         quantile=quantile,
                         max_n_points=max_n_points,
                         **kwargs)

    def _get_coordinate_extreme_predictions(self) -> List[int]: #TODO Not used?
        logger.info('Choosing datapoints with extreme predictions as examples.')
        tmp = self._get_coordinates()
        logger.debug(str(len(tmp)) + ' points with extreme prediction values chosen as parameter.')
        return tmp


if __name__ == '__main__':
    from sloth.datasets.test_sets import simple_regression_ordinal_discrete_ohe as test_task
    task = test_task(n_samples=1_000, x=2, f=0)

    dp = DataPoints(task)
    ex_pred = ExtremePredictions(task)
    ex_points = ExtremePoints(task)
    print(ex_points)
    print(ex_pred)

