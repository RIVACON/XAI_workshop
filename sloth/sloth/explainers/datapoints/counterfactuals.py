import logging
import warnings
import numpy as np
import pandas as pd
from typing import List

from sloth.features import DataType
from sloth.validation_task import ValidationTask
from sloth.explainers.datapoints.examples import Examples
from sloth.explainers.datapoints.extreme_points import DataPoints

try:
    import dice_ml
    _dice_ml_installed = True
except ImportError:
    warnings.warn('The python package dice_ml is not installed. Not all counterfactual computations can be used.')
    _dice_ml_installed = False

logger = logging.getLogger('sloth')

class _DiceWrapperModel:
    def __init__(self, task: ValidationTask, output: str=None):
        self.task = task
        self._column_names = ['']*self.task.data.shape[1]
        self._continuous_features = []
        self._discrete_features = []
        for f in self.task.input_features.values():
            if f.data_type == DataType.ONE_HOT_ENCODED:
                for i,c in enumerate(f.column):
                    self._column_names[c] = f.category_names[i]
                    self._discrete_features.append(f.category_names[i])
            elif f.data_type == DataType.ORDINAL_DISCRETE:
                self._column_names[f.column] = f.name
                self._discrete_features.append(f.name)
            else:
                self._column_names[f.column] = f.name
                self._continuous_features.append(f.name)
        self.output_name = output
        if output is None:
            if self.task.output_dim() > 1:
                raise Exception('Please specify output name if more then one output exists.')
            for f in self.task.output_features.values():
                self.output_name = f.name
        self.output_id = self.task.output_features[self.output_name].column

    def get_data(self):
        result = {self._column_names[i]: self.task.data[:, i] for i in range(len(self._column_names))}
        if len( self.task.y_pred.shape)==1:
            result[self.output_name] = self.task.y_pred
        else:
            result[self.output_name] = self.task.y_pred[:,self.output_id]
        return dice_ml.Data(dataframe=pd.DataFrame(result), continuous_features=self._continuous_features,
                            discrete_features=self._discrete_features, 
                            outcome_name=self.output_name)

    def predict(self,x):
        return self.task.predict(x[self._column_names].values)

class Counterfactual(DataPoints):
    def __init__(self,
                 task: ValidationTask,
                 examples: Examples = None,
                 max_counterfactuals=2,
                 param_method='random',
                 metric_weights=None,
                 **kwargs):

        if task.output_dim() != 1:
            raise NotImplementedError('Outputdimension != 1 not implemented yet.')

        super().__init__(validation_task=task,
                         max_counterfactuals=max_counterfactuals,
                         param_method=param_method,
                         metric_weights=metric_weights,
                         **kwargs)

        self.examples = examples
        self._setup_metric_weights()

    def _setup_metric_weights(self):
        metric_weights = {}
        for f in self.task.input_features.values():
            if f.metric_weight is not None:
                metric_weights[f.name] = f.metric_weight
            elif f.data_type == DataType.ONE_HOT_ENCODED:
                logger.debug('No metric weight for feature ' + f.name + ' defined, setting weight to 1.0.')
                metric_weights[f.name] = 1.0
            else:
                x = self.task.data[:, f.column]
                w = 1.0/np.median(np.absolute(x - np.median(x)))# Use median absolute deviation (MAD) as suggested in Wachte et.al.
                metric_weights[f.name] = w
                logger.debug('No metric weight for feature ' + f.name + ' defined, setting weight to '+str(w))
        self.metric_weights = metric_weights

    def cf_in_data(self, x: np.ndarray, 
                          target_value: float,
                          target_eps: float,
                          max_counterfactuals: int=10,
                          distance_type: str = None,
                          max_data: int = 1000,
                          seed: int = 42)->np.ndarray:
        if distance_type is None:
            distance_type = 'abs'
        # first select all points that fullfill the given value change
        if distance_type == 'abs':
            distance = np.abs(self.task.y_pred - target_value)
        elif distance_type == 'above':
            distance = self.task.y_pred-target_value
            distance[distance<0] = 2.0*target_eps
        elif distance_type == 'below':
            distance = distance = target_value-self.task.y_pred
            distance[distance<0] = 2.0*target_eps
        selection = np.where(distance<=target_eps)[0]
        if selection.shape[0] > max_data:
            selection = np.argpartition(distance, max_data)[:max_data]
        # find 'closest' points according to feature distance
        if len(selection) > 1: 
            distance = self._compute_distance_matrix(x, self.task.data[selection, :])[0, :]
            if selection.shape[0]>max_counterfactuals: 
                selection = selection[np.argpartition(distance, max_counterfactuals)[:max_counterfactuals]]
        # Now we try to find counterfactuals by setting features in current points equal to features in 
        # reference point and use the value
        cf_in_data = self.task.data[selection, :]
        cf_generated = np.copy(self.task.data[selection, :])
        for i in range(selection.shape[0]):
            features = {f.name for f in self.task.input_features.values()}
            x_orig = self.task.data[selection[i], :]
            x_new = np.copy(x_orig)
            for j in range(len(self.task.input_features)):
                dist = 1e10
                selected_feature = None
                for f in self.task.input_features.values():
                    if f.name not in features:
                        continue
                    x_new[f.column] = x[f.column]
                    d = self._compute_distance_matrix(x, x_new)[0,0]
                    if d < dist:
                        y_new = self.task.predict(x_new.reshape((1,-1)))[0]
                        #print(f.name, y_new)
                        if (y_new) + target_eps>= target_value:#np.abs(y_new-target_value) < target_eps:
                            dist = d
                            selected_feature = f
                    x_new[f.column] = x_orig[f.column]
                if selected_feature is not None:
                    #print(features)
                    features.remove(selected_feature.name) 
                    #print(features)
                    x_new[selected_feature.column] = x[selected_feature.column] 
            cf_generated[i,:]=x_new
                    

            
        return self.task.data[selection, :], cf_generated

    def compute_dice_ml(self, x: np.ndarray, target_range: List[float]):
        if not _dice_ml_installed:
            raise Exception('Module dice_ml is not installed. Either install module or use simple counterfactual search in training data.')
        model = _DiceWrapperModel(self.task)
        if self.task.problemtype == 'regression':
            model_type = 'regressor'
        else:
            model_type = 'classifier'
        dice_model = dice_ml.Model(model=model, backend='sklearn', model_type=model_type)
        dice_data = model.get_data()
        exp_random = dice_ml.Dice(dice_data, dice_model,  method=self.param_method)
        if len(x.shape)==1:
            x_ = pd.DataFrame(x.reshape((1,-1)), columns=model._column_names)
        else:
            x_ = pd.DataFrame(x, columns=model._column_names)
        dice_exp_random = exp_random.generate_counterfactuals(x_, total_CFs=self.max_counterfactuals,
                                                      desired_range=target_range,
                                                      #desired_class="opposite", 
                                                      verbose=False)
        return dice_exp_random.cf_examples_list[0].final_cfs_df[model._column_names].values
        #return dice_exp_random
        

    def get_table(self, point: np.ndarray, counterfactuals: np.ndarray)->pd.DataFrame:
        result = {'Feature': [f.name for f in self.task.input_features.values()] + ['prediction', 'distance']}
        for i in range(counterfactuals.shape[0]):
            result_ = []
            for f in self.task.input_features.values():
                result_.append(f._get_value_for_table(counterfactuals[i,:]))
            result_.append(self.task.predict(counterfactuals[i,:].reshape((1,-1)))[0])
            result_.append(self._compute_distance_matrix(counterfactuals[i,:], point)[0,0])
            result['cf_'+str(i)] =result_
        result_ = []
        for f in self.task.input_features.values():
            result_.append(f._get_value_for_table(point))
        result_.append(self.task.predict(point.reshape((1,-1)))[0])
        result_.append(0.0)
        result['Example'] = result_
        column_order = ['Feature', 'Example'] + ['cf_'+str(i) for i in range(counterfactuals.shape[0])]
        return pd.DataFrame(result)[column_order]

    def _compute_distance_matrix(self, x1:np.ndarray,x2: np.ndarray):
        x1_ = x1
        if len(x1.shape) == 1:
            x1_ = x1.reshape((1,-1))
        x2_ = x2
        if len(x2.shape) == 1:
            x2_ = x2.reshape((1,-1))
        if x1_.shape[0] > x2_.shape[0]:
            return self._compute_distance_matrix(x2_,x1_)
        result = np.zeros((x1_.shape[0], x2_.shape[0]))
        for f in self.task.input_features.values():
            weight = self.metric_weights[f.name]
            for i in range(x1_.shape[0]):
                result[i,:] += weight*f.metric(x1_[i,f.column],x2_[:,f.column])
            # if f.data_type == DataType.ONE_HOT_ENCODED:
            #     for i in range(x1_.shape[0]):
            #         result[i,:] += weight*(x1_[i,f.column]&x2_[:,f.column])
            # else:
            #     for i in range(x1_.shape[0]):
            #         result[i,:] += weight*(np.abs(x1_[i,f.column]-x2_[:,f.column]))
        return result
    


if __name__ == "__main__":
    import sloth

    validation_task = sloth.datasets.test_sets.simple_classification_ordinal(n_samples=1_000, x=0, f=0)

    ex = Examples(validation_task)
    cf = Counterfactual(validation_task, max_counterfactuals=7)
    # cf.compute_dice_ml(validation_task.data[0, :], target_range=None) #TODO was macht das
   
