from typing import Dict, Union, Tuple, List
import logging
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sloth.features import DataType, FeatureDescription, OneHotEncodedFeatureDescription
from sloth.validation_task import ValidationTask
from sloth.exceptions import WrongDataType
from sloth.explainers.global_explainers.base_global_explainer import BaseGlobalExplainer, BaseGlobalExplanation

logger = logging.getLogger('sloth')

class ALE(BaseGlobalExplainer):
    def __init__(self,
                 task: ValidationTask,
                 min_value=None,
                 max_value=None,
                 n_steps=10,
                 **kwargs
                 ):
        """Accumulated local effects.

        Accumulated local effects give an insight to the global model behaviour w.r.t. a single feature.
        For strongly correlated features, partial dependence plots may be misleading due to the influence of
        unrealistic sampled points. ALE tries to overcome this issue by dividing the domain of the feature of interest
        

        Args:
            validation_task (ValidationTask): Validation task to be analyzed.
            min_value (dict): Dictionary of minimum values for each feature. If not set, the minimum value is set to the minimum value in the task data.
            max_value (dict): Dictionary of maximum values for each feature. If not set, the maximum value is set to the maximum value in the task data.
            n_steps (int): Number of steps used to discretize domain in coordinate of interest where the
                domain boundaries are determined by the permitted_range member of the respective feature (if set)
                or otherwise by the maximum and minimum of the feature
                values in the validation task data.
        Examples
        --------
            >>> import sloth
            >>> ale = sloth.explainers.global_explainers.ALE(validation_task, n_steps=50)
            >>> ale.explain()
            {'age': {'boundaries': array([0.00204453, 0.02012188, 0.03819923, 0.05627658, 0.07435393,
            0.09243128, 0.11050863, 0.12858598, 0.14666333, 0.16474068,
            0.18281802, 0.20089537, 0.21897272, 0.23705007, 0.25512742,
            0.27320477, 0.29128212, 0.30935947, 0.32743682, 0.34551417,
            0.36359152, 0.38166887, 0.39974622, 0.41782357, 0.43590091,
            0.45397826, 0.47205561, 0.49013296, 0.50821031, 0.52628766,
            0.54436501, 0.56244236, 0.58051971, 0.59859706, 0.61667441,
        """
        if task.output_dim() != 1:
            raise NotImplementedError('Outputdimension != 1 not implemented yet.')
        if task.problemtype != 'regression':
            raise NotImplementedError("Accumulated local effects is only implemented for tasks in the case of regression.")
            #TODO Not Implemented Error or Value Error
            #Optional case for classification by using ALE line for each predicting class instead of each feature
            #Furter information on https://docs.seldon.io/projects/alibi/en/latest/examples/ale_classification.html
        super().__init__(task, min_value=min_value, max_value=max_value, n_steps=n_steps, **kwargs)

    def _explain(self)->BaseGlobalExplanation:
        """Computes the accumulated local effects for all ordinal features in the validation task.

        Returns:
            BaseGlobalExplanation: Global explanation for the validation task.
        """
        # result = BaseGlobalExplanation(self.task, self._hashkey, 'ALE plot')
        result = ALEExplanation(self.task, self._hashkey, 'ALE plot')
        for f in self.task.input_features.values():
            if f.data_type != DataType.ORDINAL.value: # only applicable for ordinal features
                continue
            if self.min_value is not None and f.name in self.min_value.keys():
                min_value = self.min_value[f.name]
            else:
                min_value = None
            if self.max_value is not None and f.name in self.max_value.keys():
                max_value = self.max_value[f.name]
            else:
                max_value = None
            self.__explain(f, result, min_value=min_value, max_value=max_value)
        return result
    
    def __explain(self,
                feature: str, 
                result: BaseGlobalExplanation,
                min_value, max_value):
        if isinstance(feature, str):
            feature = self.task.input_features[feature]
        WrongDataType.check_equality(feature, DataType.ORDINAL)
        if min_value is None:
            min_value = self.task.data[:, feature.column].min()
        if max_value is None:
            max_value = self.task.data[:, feature.column].max()
        
        boundaries = np.linspace(min_value, max_value, self.n_steps)
        projection = 0.5*(boundaries[1:]+boundaries[:-1])
        projected_values = np.empty(projection.shape)
        for i in range(projection.shape[0]):
            selection = (self.task.data[:, feature.column] >= boundaries[i]) & ((self.task.data[:, feature.column] <= boundaries[i + 1]))
            x = np.copy(self.task.data[selection, :])
            if x.shape[0] < 1:
                projected_values[i] = None
                continue  
            x[:,feature.column] = boundaries[i] 
            y_left = self.task.predict(x)
            x[:,feature.column] = boundaries[i+1]
            y_right = self.task.predict(x)
            y = (y_right-y_left)#/(boundaries[i+1]-boundaries[i])
            if len(y.shape)==1:
                y = y.reshape((-1,1))
            projected_values[i] = y.mean()
        if feature.inverse_transform is not None:
            projection = feature.inverse_transform(projection)
        result.add_result(feature.name, projection, projected_values)


class ALEExplanation(BaseGlobalExplanation):
    def __init__(self, task: ValidationTask, explainer_hash: str, method: str):
        # Comment is inherited from parent class
        __doc__ = BaseGlobalExplanation.__doc__

        super().__init__(task, explainer_hash, method)

if __name__ == "__main__":

    import sloth
    #validation_task = sloth.datasets.test_sets.simple_classification_ordinal(n_samples=1_000, x=0, f=0)
    validation_task = sloth.datasets.test_sets.simple_regression_ordinal_discrete_ohe(n_samples=1000, x=0, f=0)
    expl = ALE(validation_task)
    data = validation_task.data
    x = data[0]
    df = expl.explain()
    help(df)
    df.plot()
