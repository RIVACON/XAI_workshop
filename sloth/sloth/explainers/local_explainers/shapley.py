import corr_shap
from corr_shap.sampling.SamplingStrategy import SamplingStrategy
from shap.utils._legacy import IdentityLink
import typing
import numpy as np

from sloth.explainers.local_explainers.base_local_explainer import BaseLocalExplainer
from sloth.validation_task import ValidationTask

class ShapExplainer(BaseLocalExplainer):
    def __init__(self, task: ValidationTask,
                 link='identity', #Only 'logit' or 'identity'. Pass iml.Link objekt otherwise
                 sampling='default',
                 sampling_kwargs={},
                 **kwargs
                 ):
        super().__init__(task=task,
                         link=link,
                         sampling=sampling,
                         sampling_kwargs=sampling_kwargs,
                         **kwargs)
        self.explainer = corr_shap.CorrExplainer(model=task.predict,
                                                 data=task.data,
                                                 link=self.link,
                                                 sampling=self.sampling,
                                                 sampling_kwargs=self.sampling_kwargs,
                                                 feature_names=self.task.get_input_names())

    def _explain(self, x: np.ndarray):
        return self.explainer(x)


if __name__ == "__main__":
    import sloth
    validation_task = sloth.datasets.test_sets.simple_classification_ordinal(n_samples=1_000, x=0, f=0)
    shapExpl = ShapExplainer(validation_task)
    data = validation_task.data
    x = data[0]
    df = shapExpl.explain(x=x)
    print(df)
