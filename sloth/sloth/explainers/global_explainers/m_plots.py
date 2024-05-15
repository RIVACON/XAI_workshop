import logging
import numpy as np

from typing import List, Union, Dict
from sloth.validation_task import ValidationTask
from sloth.features import FeatureDescription, DataType
from sloth.exceptions import WrongDataType
from sloth.explainers.global_explainers.base_global_explainer import BaseGlobalExplainer,  BaseGlobalExplanation

logger = logging.getLogger('sloth')

class MarginalPlots(BaseGlobalExplainer):
    def __init__(self,
                 task: ValidationTask,
                 n_steps=10,
                 **kwargs):
        """Marginal Plot explainer.

        Partial Dependence Plots rely on the assumption that the features being used to compute the marginal distribution are independent of the other features. If this assumption is violated, the resulting plot may be distorted and based on unrealistic data points. To mitigate this issue, one possible solution is to integrate over the conditional distribution of the other features. This is where Marginal Plots enter the game. They plot the marginal function
        
        .. math::
        
            f_S(x_S) := \int f(x_S,x_{-S})dP(x_{-S}\mid X_S)
        
        where :math:`S` is an index set (usual with one or two elements) and :math:`P(x_{-S}\mid X_S)` is the conditional density w.r.t. to fixed :math:`X_S`.

        To approximate the conditional density for a certain :math:`x_S`, we first compute bins :math:`I_k` that contain the same number of feature points. Then, we compute the conditional density for each bin :math:`I_k` as the average of the model values over all points in the respective bin. 

        Args:
            task (ValidationTask): The validation task for which the explainer is computed.
            n_steps (int, optional): Number of bins. Defaults to 100.

        Raises:
            NotImplementedError: Raised when the validation task has more then one output dimension.

        Examples
        --------
            >>> import sloth
            >>> validation_task = sloth.datasets.credit_default.get(dataset=0, model=0)
            >>> mp = sloth.explainers.MarginalPlots(validation_task, n_steps=50)
            >>> mp_explanation = pdp.explain()
            >>> mp_explanation.plot()

        """
        if task.output_dim() != 1:
            raise NotImplementedError('Outputdimension != 1 not implemented yet.')

        super().__init__(task, n_steps=n_steps, **kwargs)
        
    def _compute_ordinal(self, y: np.ndarray, feature: FeatureDescription):
        WrongDataType.check_equality(feature, DataType.ORDINAL)
        x = self.task.data[:, feature.column]
        bucket_counts, bin_edges = np.histogram(x, bins=self.n_steps)
        mean_values, bin_edges = np.histogram(x, bins=self.n_steps, weights=y)
        mean_values[bucket_counts>0] = mean_values[bucket_counts>0]/bucket_counts[bucket_counts>0]
        return bin_edges, mean_values, bucket_counts
    
    def _explain(self) -> BaseGlobalExplanation:
        result = BaseGlobalExplanation(self.task, self._hashkey, 'marginal plot')
        for f in self.task.input_features.values():
            if f.data_type != DataType.ORDINAL.value: # only applicable for ordinal features
                continue
            self.__explain(f, result)
        return result
    
    def __explain(self, feature: FeatureDescription, result: BaseGlobalExplanation):
        WrongDataType.check_equality(feature, DataType.ORDINAL)
        y_pred = self.task.y_pred #self.task.predict(self.task.data)
        bin_edges, mean_values, bucket_counts = self._compute_ordinal(y_pred,feature)
        bucket_midpoints = 0.5*(bin_edges[1:]+bin_edges[:-1])
        y = np.interp(bucket_midpoints, bucket_midpoints[bucket_counts>0], mean_values[bucket_counts>0])
        if feature.inverse_transform is not None:
            bucket_midpoints = feature.inverse_transform(bucket_midpoints)
        result.add_result(feature.name, bucket_midpoints, y)
        
    # def plot(self, n_plots: tuple=None, include_rug = True, 
    #          include_points = True, features: Union[str,List[str]]=None,
    #          new_figure:bool=True, label:str=None):
    #     if label is None:
    #         label='MP'
    #     #y_pred = self.task.predict(self.task.data)
    #     num_plot = 1
    #     output = next(iter(self.task.output_features.values()))
    #     if features is None:
    #         features = self.task.input_features.values()
    #     elif isinstance(features, str):
    #         features = [self.task.input_features[features]]
    #     else:
    #         features = [self.task.input_features[f] for f in features]
    #     for f in features:
    #         if f.data_type == DataType.ORDINAL:
    #             cache = self.explain(f)
    #             projection, projected_values = cache['projection'], cache['projected_values']
    #             if n_plots is not None:
    #                 plt.subplot(n_plots[0], n_plots[1], num_plot)
    #             else:
    #                 if new_figure:
    #                     plt.figure()
    #             num_plot += 1
    #             plt.plot(projection, projected_values, '-', linewidth=3.0, label=label)
    #             if include_rug:
    #                 if f.inverse_transform is not None:
    #                      sns.rugplot(f.inverse_transform(self.task.data[:, f.column]), lw=1, alpha=.1)
    #                 else:
    #                     sns.rugplot(self.task.data[:, f.column], lw=1, alpha=.1)

    #             #plt.hist(self.data[:,f.column])
    #             plt.xlabel(f.name)
    #             plt.ylabel(output.name)
    #             plt.legend()
    #             plt.title('Marginal Plot')


class MarginalPlotsExplanation(BaseGlobalExplanation):
    def __init__(self, task: ValidationTask, explainer_hash: str, method: str):
        # Comment is inherited from parent class
        __doc__ = BaseGlobalExplanation.__doc__

        super().__init__(task, explainer_hash, method)

if __name__ == "__main__":
    import sloth

    validation_task = sloth.datasets.test_sets.simple_classification_ordinal(n_samples=1_000, x=0, f=0)
    mp = MarginalPlots(validation_task)
    expl = mp.explain()
    expl.plot()
