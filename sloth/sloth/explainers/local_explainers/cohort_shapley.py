import logging
import warnings
from typing import Tuple, List
import numpy as np

from sloth.validation_task import ValidationTask
from sloth.explainers.local_explainers.base_local_explainer import BaseLocalExplainer

import sloth.datasets.credit_default2 as biased_dataset
from sloth.features import DataType
from sloth.explainers.base_explainer import BaseExplanation

logger = logging.getLogger('sloth')

try:
    import shap 
    _shap_installed = True
except ImportError:
    warnings.warn('The python package shap is not installed. Some visualization methods for the ChortSHap may not be working.')
    _shap_installed = False

try:
    import sys
    sys.path.append(r'C:\Users\dsoellheim\PycharmProjects\cohortshapley')
    from cohortshapley import cohortshapley
    from cohortshapley import similarity
    cohortshapley_installed = True
except ModuleNotFoundError:
    cohortshapley_installed = False
    logger.info('Python package cohortshapley not installed, cannot compute cohort shapley values without this package.')
    warnings.warn('Python package cohortshapley not installed, cannot compute cohort shapley values without this package.')

class ShapExplanation(BaseExplanation):
    """ Wrapper for SHAP functionalities (plotting) """

    def __init__(self, shap_values: np.ndarray, 
                 data: np.ndarray,
                 task: ValidationTask,
                 explainer_hash: str):
        super().__init__(task, explainer_hash)
        self.shap_values = shap_values
        self.data = data

    def plot_beeswarm(self, **kwargs): 
        """ Beeswarm plot 
        
        default arguments: 
        max_display=10, 
        order=shap.Explanation.abs.mean(0), 
        clustering=None, 
        cluster_threshold=0.5, 
        color=None, 
        axis_color='#333333',
        alpha=1, 
        show=True, 
        log_scale=False, 
        color_bar=True, 
        plot_size='auto', 
        color_bar_label='Feature value'
        """
        
        explanation = shap.Explanation(self.shap_values, data=self.data, feature_names=self._task.get_input_names()) #TODO implement a better way in validation task to retrieve values
        shap.plots.beeswarm(explanation, **kwargs)

    def plot_bar(self, row: int=None, **kwargs):
        """ Bar plot

        default arguments: 
        max_display=10, 
        order=shap.Explanation.abs, 
        clustering=None, 
        clustering_cutoff=0.5, 
        merge_cohorts=False, 
        show_data='auto', 
        show=True

        Args:
            row (int, optional): Row of shap values, for local bar plot. Defaults to None.
        """

        explanation = shap.Explanation(self.shap_values, data=self.data, feature_names=self._task.get_input_names())
        if row is None: #global bar plot
            shap.plots.bar(explanation, **kwargs)
        else: #local bar plot
            shap.plots.bar(explanation[row], **kwargs)

class CohortShapleyValues(BaseLocalExplainer):

    def __init__(self,
                 task: ValidationTask,
                 similarity_ratio=0.1,
                 bool_error=False,
                 **kwargs):

        super().__init__(task,
                         similarity_ratio=similarity_ratio,
                         bool_error=bool_error,
                         **kwargs)

    @staticmethod
    def _similarity(input_features:dict, names:List[str])->callable:
        def __similarity(data: np.ndarray, subject: np.ndarray, vertex: np.ndarray)->np.ndarray:
            """Returns a vector of 1 and zero indicating whether the respective datapoint 
            (of the validation task) is similar to the subject (according to the coordinates included in vertex).

            Args:
                subject (np.ndarray): The subject for which the similarity is computed.
                vertex (np.ndarray): The coordinates of the subject that are used for the similarity computation.

            Raises:
                NotImplementedError: _description_

            Returns:
                np.ndarray: Array of 1 and 0 (length of all datapoints in the validation task) indicating whether the respective datapoint is similar to the subject.
            """
            ccond = np.ones(data.shape[0])
            for i in range(vertex.shape[-1]):
                if vertex[i] == 0:
                    continue
                feature = input_features[names[i]]
                if feature.data_type == DataType.ORDINAL:
                    ccond = np.logical_and(ccond, feature.similarity(subject[i],data[:,i]))
                else:
                    ccond = np.logical_and(ccond, np.equal(subject[i], data[:,i]))
            return ccond
        return __similarity
    
    def _explain(self, x: np.ndarray) -> ShapExplanation: 
        if not cohortshapley_installed:
            raise ModuleNotFoundError('Python package cohortshapley not installed, cannot compute cohort shapley values without this package.')
        if self.task.output_dim() != 1:
            raise NotImplementedError('Cohort Shapley Variance is only implemented for output dimension 1.')
        if len(x.shape)==1:
            x = x.reshape(1,-1)
        #if True:
        data = np.concatenate((self.task.data, x)) # [:,:-1] if there is an output feature
        y = np.concatenate((self.task.y_pred, self.task.predict(x)))
        if self.bool_error:
            id_col = self.task.output_features['default probability'].column
            true_data = self.task.data[:,id_col]
            y = np.concatenate((self.task.y_pred-true_data, self.task.predict(x)))
        subject_id = [self.task.data.shape[0] + i for i in range(x.shape[0])]
        # compute cohort Shapley
        similarity.ratio = self.similarity_ratio
        cs_obj = cohortshapley.CohortShapley(None, 
                                            similarity.similar_in_distance_cutoff, 
                                            subject_id, data, y=y)
        cs_obj.compute_cohort_shapley()
        return ShapExplanation(cs_obj.shapley_values, x, self.task, self._hashkey)
        # else:
        #     data, names = self.task.get_data_ohe_as_ordinal(np.concatenate((self.task.data, x)))
        #     y =  np.concatenate((self.task.y_pred, self.task.predict(x)))
        #     subject_id = [self.task.data.shape[0] + i for i in range(x.shape[0])]
        #     similarity_ = CohortShapleyValues._similarity(self.task.input_features, names)
        #     cs_obj = cohortshapley.CohortShapley(None, 
        #                                         similarity_, 
        #                                         subject_id, data, y=y)
        #     cs_obj.compute_cohort_shapley()    
        #     return ShapExplanation(cs_obj.shapley_values, x, self.task, self._hashkey)

if __name__=='__main__':
    
    import sloth
    from sloth.explainers.local_explainers.cohort_shapley import ShapExplanation


    #validation_task = sloth.datasets.test_sets.simple_regression_ordinal_discrete_ohe(n_samples=10_000, x=0, f=0)
    #validation_task = sloth.datasets.test_sets.simple_classification_ordinal(n_samples=10_000, x=0, f=0) 
    validation_task = biased_dataset.get()
    cohort_shap_values = sloth.explainers.local_explainers.CohortShapleyValues(validation_task)

    cs = cohort_shap_values.explain(validation_task.data[:100,:])
    #cs.plot_beeswarm()
    cs.plot_bar()
