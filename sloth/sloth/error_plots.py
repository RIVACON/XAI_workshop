import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import sloth
from sklearn import  tree
from sloth.features import DataType, FeatureDescription
from sloth.validation_task import ValidationTask
from sloth.exceptions import FeatureDoesNotExist

def plot_vs_feature(x,error, feature_name: str, 
                          task: ValidationTask):
    FeatureDoesNotExist.check(task.input_features, feature_name)
    feature = task.input_features[feature_name]
    output_name = next(iter(task.output_features.keys()))
    if feature.data_type == DataType.ORDINAL.value:
        plt.plot(x[:,feature.column], error, '.', alpha=0.005)
        #model = tree.DecisionTreeRegressor(min_samples_leaf=100).fit(x[:,feature.column].reshape((-1,1,)),error)
        #error_pred = model.predict(x[:,feature.column].reshape((-1,1,)))
        #plt.plot(x[:,feature.column], error_pred, '.', alpha=0.1)
        #data = np.concatenate([x,error.reshape((-1,1))], axis=1)
        #sn.scatterplot(data, x = feature.column, y= data.shape[1]-1, hue=data.shape[1]-1)
        plt.xlabel(feature.name)
        plt.ylabel('difference of ' + output_name + ' between target and prediction')
        #plt.ylim(-0.0,0.01)
        pass
    elif feature.data_type == DataType.ORDINAL_DISCRETE.value:
        x_ = x[:,feature.column]
        data = []
        x_ticks=[]
        for k in np.unique(x_):
            data.append(error[x_==k])
            x_ticks.append(k)
        plt.violinplot(data, positions=x_ticks, widths=0.1, vert=True, showmeans=True)
        plt.xticks(ticks=x_ticks)#,labels=x_label)
        plt.ylabel('difference of ' + output_name + ' between target and prediction')
        plt.xlabel(feature.name)
        pass