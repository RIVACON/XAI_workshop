import logging
import warnings
import numpy as np
import pandas as pd
from sloth.features import DataType, FeatureDescription
from sloth.explainers.local_explainers.base_local_explainer import BaseLocalExplainer
from sloth.validation_task import ValidationTask

try:
    import alibi
    alibi_installed = True
except ModuleNotFoundError:
    alibi_installed = False
try:
    import sys
    from anchor import anchor_tabular
    anchor_installed = True
except ModuleNotFoundError:
    anchor_installed = False

logger = logging.getLogger('sloth')

if not alibi_installed:
    logger.info('Python package alibi not installed, cannot compute anchors with this package.')
    warnings.warn('Python package alibi not installed, cannot compute anchors with this package.')

if not anchor_installed:
    logger.info('Python package anchor-exp not installed, cannot compute anchors with this package.')
    warnings.warn('Python package anchor-exp not installed, cannot compute anchors with this package.')

if not(anchor_installed and alibi_installed):
    logger.info('Python packages alibi and anchor-exp not installed, cannot compute anchors with this package.')
    warnings.warn('Python packages alibi and anchor-exp not installed, cannot compute anchors with this package.')


class Anchors(BaseLocalExplainer):
    def __init__(self,
                 task: ValidationTask,
                 **kwargs):
        super().__init__(task, **kwargs)

        feature_names = []  # build feature names in order of their respective columns
        for f in self.task.input_features.values():
            try:
                entry = (min(f.column), f.name)
            except:
                entry = (f.column, f.name)
            feature_names.append(entry)
        feature_names.sort()
        self.feature_names = [f[1] for f in feature_names]

        self.ohe = False #TODO ist das mit dem one hot encoded richtig so?
        self.categorical_names = {}
        for f in self.task.input_features.values():
            if f.data_type == DataType.ONE_HOT_ENCODED:
                self.ohe = True
                self.categorical_names[f.column[0]] = [c for c in f.category_names]
        #TODO Soll im default alibi oder exp ausgewählt werden?

#TODO diese _explain ist komplett überschrieben, oder?


    # def _explain(self, x: np.ndarray)->pd.DataFrame:
    #     """
    #     Compute Anchors a single instance or a batch of instances.
    #
    #     Args:
    #         x (np.ndarray): A single point or a set of pints (2d array where each row represents a single point) for which anchors are computed.
    #
    #     """
    #     if len(np.shape(x)) == 1:
    #         x = [x]
    #
    #     predict = self.task.predict
    #
    #     result = {'anchor':[], 'precision':[], 'coverage':[], 'point':[], 'target':[]}
    #     for p in x:
    #         p_ = p.reshape(1, -1)
    #         y_ref = self.task.predict(p_)
    #
    #         if self.task.problemtype == 'regression':  # TODO deal with this in task/model?
    #             def predict(x_):
    #                 y = self.task.predict(x_)
    #                 return np.abs(y - y_ref)<self.epsilon_output  # TODO y-y_ref always 0?
    #         result_p = self._explain(p_, predict)
    #         result['anchor'].append(result_p['anchor'])
    #         result['precision'].append(result_p['precision'])
    #         result['coverage'].append(result_p['coverage'])
    #         result['point'].append(p)
    #         if self.task.problemtype == 'regression':
    #             result['target'].append(self.task.y_pred[p]) # TODO p is not an index any more
    #         else:
    #             result['target'].append(y_ref)
    #     return pd.DataFrame(result)


if __name__ == "__main__":
    import sloth

    validation_task = sloth.datasets.test_sets.simple_classification_ordinal(n_samples=1_000, x=0, f=0)
    # expl = sloth.Anchors(validation_task, method='anchorsexp')
    expl_ali = sloth.AnchorsAlibi(validation_task)
    expl_exp = sloth.AnchorsExp(validation_task)
    data = validation_task.data
