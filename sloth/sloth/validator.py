import logging
from typing import Union
from sloth.validation_task import ValidationTask
from sloth.explainers.local_explainers.ice import ICE
from sloth import PDP
from sloth.explainers.datapoints.examples import Examples
from sloth.explainers.utils import UtilsClass
    


# class Validator_Parameter(Parameter):
#
#     def __init__(self, plot_param: Union[dict, PlotParameter]=None,
#                  ice_param: dict=None,
#                  #pdp_param: Union[dict, PDP_Parameter]=None,
#                  examples_param: Union[dict, Examples_Parameter]=None):
#         self.plot_param = PlotParameter.create(plot_param)
#         self.ice_param = ice_param
#         self.pdp_param = None #PDP_Parameter.create(pdp_param)
#         self.examples_param = Examples_Parameter.create(examples_param)

#     def _to_dict(self)->dict:
#         return {'ice_param': self.ice_param.to_dict(), 'pdp_param': self.pdp_param.to_dict(), 'examples_param': self.examples_param.to_dict()}
#
class Validator:
    def __init__(self, validation_task: ValidationTask):
        self.task = validation_task

        parameter_type = 'sub_plot'
        self.plot_param = UtilsClass.set_config(parameter_type)
        self.ice_param = None
        self.pdp_param = None
        # self.examples_param = Examples_Parameter.create(examples_param)

        self.examples = Examples(validation_task)
        

    def warning(self, msg):
        logging.warning(msg)

    def pdp(self, features=None):
        pdp = PDP(self.task, self.pdp_param)
        pdp.plot(features=features)

    def ice(self):
        ice = ICE(self.task, self.ice_param, self.examples, self.warning)
        ice.plot()

    def ice_cluster(self, features=None):
        ice = ICE(self.task, self.ice_param, self.examples)
        ice.plot_cluster()
    
    def examples_hiplot(self):
        self.examples.hiplot()


if __name__ == '__main__':
    from sklearn.datasets import load_diabetes
    from sklearn import linear_model
    import sloth
    from sloth.validator import Validator

    x, y = load_diabetes(as_frame=True, return_X_y=True)
    lin_regr = linear_model.LinearRegression().fit(x, y)

    input_features = [sloth.OrdinalFeatureDescription(name=name) for name in x.columns if name != 'sex']  # all feature ordinal except sex
    input_features.append(sloth.DiscreteOrdinalFeatureDescription(name='sex'))
    output_feature = sloth.OrdinalFeatureDescription(name='disease progression', column=0)
    validation_task = sloth.ValidationTask(input_features=input_features, output_features=output_feature,
                                           data=x, predict=lin_regr.predict, problemtype='regression')

    validator_object = sloth.validator.Validator(validation_task)
    validator_object.pdp()
