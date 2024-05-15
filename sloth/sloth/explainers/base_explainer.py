import logging
from sloth.validation_task import ValidationTask
from sloth.explainers.utils import UtilsClass

logger = logging.getLogger("sloth")


class BaseExplanation:

    def __init__(self, task: ValidationTask, explainer_hash: str):
        self._task = task
        self._explainer_hash = explainer_hash

    def plot(self, **kwargs):
        raise Exception("Plotting is not provided for this kind of explanation.")

    def df(self):
        raise Exception(
            "A DataFrame summarizing the results is not provided for this kind of explanation."
        )


class BaseExplainer:
    def __init__(self, task: ValidationTask, **kwargs):
        self.task = task
        if kwargs is not None:
            for k, v in kwargs.items():
                setattr(self, k, v)
        self._hashkey = self.__hashkey(kwargs)


    def __hashkey(self, kwargs) -> str:
        tmp = {"task": self.task._hashkey, "method": type(self).__name__, "params": kwargs}
        return UtilsClass.hash_for_dict(tmp)


if __name__=='__main__':
    from sklearn.datasets import load_diabetes
    from sklearn import linear_model
    import sloth

    x, y = load_diabetes(as_frame=True, return_X_y=True)
    lin_regr = linear_model.LinearRegression().fit(x, y)

    input_features = [sloth.OrdinalFeatureDescription(name=name) for name in x.columns if
                      name != 'sex']  # all feature ordinal except sex
    input_features.append(sloth.DiscreteOrdinalFeatureDescription(name='sex'))

    output_feature = sloth.OrdinalFeatureDescription(name='disease progression', column=0)

    validation_task = sloth.ValidationTask(input_features=input_features, output_features=output_feature,
                                           data=x, predict=lin_regr.predict, problemtype='regression')
    ex = BaseExplainer(validation_task, 'pdp')
