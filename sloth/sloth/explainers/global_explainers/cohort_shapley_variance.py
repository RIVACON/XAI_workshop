import logging
import warnings
from typing import Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sloth.features import DataType

logger = logging.getLogger('sloth')
import sys


try:
    sys.path.append('C:/Users/dsoellheim/PycharmProjects/cohortshapley/')
    from cohortshapley import varianceshapley
    from cohortshapley import similarity
    cohortshapley_installed = True
except ModuleNotFoundError:
    cohortshapley_installed = False
    logger.info('Python package cohortshapley not installed,'
                'cannot compute cohort shapley values without this package.')
    warnings.warn('Python package cohortshapley not installed,'
                  'cannot compute cohort shapley values without this package.')

from sloth.validation_task import ValidationTask
from sloth.explainers.global_explainers.base_global_explainer import BaseGlobalExplainer, BaseGlobalExplanation


class CohortShapleyVarianceExplanation(BaseGlobalExplanation):
    def __init__(self, task: ValidationTask,
                 explainer_hash: str,
                 method: str,
                 feature_names:List[str], shapley_variance:List[str]):
        # Comment is inherited from parent class
        __doc__ = BaseGlobalExplanation.__doc__

        super().__init__(task, explainer_hash, method)
        self.feature_names = feature_names
        self.shapley_variance = shapley_variance
        self.shapley_variance_relative = shapley_variance/np.sum(shapley_variance)

    def df(self)->pd.DataFrame:
        """Returns the shapley variance values as a pandas dataframe.

        Returns:
            pd.DataFrame: Shapley variance values
        """
        return pd.DataFrame({'feature':self.feature_names, 'shapley_variance':self.shapley_variance, 'shapley_variance_relative':self.shapley_variance_relative})
    
    def plot(self, show=True):
        #TODO Useful to choose subset of feature to plot?
        plt.pie(self.shapley_variance_relative, labels=self.feature_names, autopct='%1.1f%%', shadow=True, startangle=90)
        plt.legend()
        plt.title('Feature Cohort Shapley Variance')
        if show:
            plt.show()


class CohortShapleyVariance(BaseGlobalExplainer):
    def __init__(self, 
                task: ValidationTask,
                 similarity_bins=10,
                **kwargs):
        """Cohort Shapley Variance explainer.

        The cohot shapley method is a global explainer that computes the shapley values for each feature on the
        given data set only, avoiding sampling of new data points. This may be of of interest if the data is
        strongly correlated avoiding the creation of very unrealistic data sets that may be created by sampling.
        This class uses the cohort shapley package that must be installed separately and can be find `here <https://github.com/cohortshapley/cohortshapley>`_.
        
        Learn more in the :ref:`User Guide<User-Guide-Global-Methods>`.

        Args:
            task (ValidationTask): Validation task to be analyzed.
            similarity_bins (int): Number of bins created for each feature.

        Raises:
            NotImplementedError: If output dimension is not 1.

        See Also
        --------
            :ref:`CohortShapleyValues` : Cohort Shapley to compute individual local explanations.

        Examples
        --------
            >>> import sloth
            >>> validation_task = sloth.datasets.test_sets.simple_regression_ordinal(n_samples=1_000, x=0, f=0)
            >>> cohort_shap_variance = sloth.explainers.global_explainers.CohortShapleyVariance(validation_task, similarity_bins=50)
            >>> print(cohort_shap_variance.explain())
            (array([0.05347124, 0.05451855, 0.0041261 ]), ['x_1', 'x_2', 'x_3'])
        """
        super().__init__(task, similarity_bins=similarity_bins, **kwargs)

    def explain(self)->CohortShapleyVarianceExplanation:
        return super().explain()
    
    def _explain(self)->Tuple[np.ndarray,List[str]]:
        feature_names = list(self.task.input_features.keys())
        bins = []
        for f in feature_names:
            feature = self.task.input_features[f]
            if feature.data_type == DataType.ORDINAL:
                cols = (feature.column,)
                bins.append(similarity.binning(self.task.data[:,cols], bins=self.similarity_bins)[0])
            elif feature.data_type == DataType.ORDINAL_DISCRETE:
                values = np.unique(self.task.data[:,feature.column])
                n_categories = (values).shape[0]
                bins_ = np.full((self.task.data.shape[0],1), -1, dtype='int') #we initialize with negative value to see if there are missing values at the end
                for i in range(n_categories):
                    bins_[np.where(self.task.data[:,feature.column]==values[i])[0]] = i+1
                bins.append(bins_)
            else:
                n_categories = len(feature.category_names)
                bins_ = np.full((self.task.data.shape[0],1), -1, dtype='int')
                for i in range(n_categories):
                    index = np.where(self.task.data[:,feature.column[i]]==1)[0]
                    bins_[index,:] = i+1 #we initialize with negative value to see if there are missing values at the end
                if bins_.min()<0:
                    raise Exception('Missing values in one-hot encoded feature')
                bins.append(bins_)
        bins = np.concatenate(bins,axis=1)
        vs_values = varianceshapley.VarianceShapley(self.task.y_pred, bins)
        return CohortShapleyVarianceExplanation(self.task, self._hashkey, 'Cohort Shapley Variance', feature_names, vs_values)
        

if __name__ == '__main__':
    import sloth
    # validation_task = sloth.datasets.test_sets.simple_regression_ordinal_discrete_ohe(n_samples=1_000, x=0, f=0)
    validation_task = sloth.datasets.test_sets.simple_regression_ordinal_discrete(n_samples=1_000, x=0, f=0)
    cohort_shap_variance = sloth.explainers.global_explainers.CohortShapleyVariance(validation_task)
    expl = cohort_shap_variance.explain()
    expl.plot()

