from typing import Dict, Union, Tuple, List
import logging
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sloth.features import DataType, FeatureDescription, OneHotEncodedFeatureDescription
from sloth.validation_task import ValidationTask
from sloth.exceptions import WrongDataType
from sloth.explainers.global_explainers.base_global_explainer import BaseGlobalExplainer, BaseExplanation, BaseGlobalExplanation

logger = logging.getLogger('sloth')

class PDP(BaseGlobalExplainer):
    def __init__(self, 
                task: ValidationTask,
                n_steps=100,
                **kwargs):
        """One dimensional partial dependence plot for ordinal features.

        It approximates for a given fixed feature ..math:`i` the marginal effect defined by 
        
        .. math::
        
            f_S(x_i) := \\int f(x_i,x_{-i})dP(x_{-i})
        
        where :math:`x_{-i}` is the vector of all features except :math:`i` and :math:`P` is the distribution of the data.
        The approximation is done by simple Monte Carlo integration over the datapoints of the respective validation task
        
        .. math::
        
            f_S(x_i) \\approx\\frac{1}{n}\\sum_k \\int f(x_i,x^k_{-i})
        
        where :math:`x^k_{-i}` is the k-th datapoint of the validation task without the i-th feature.
        
        Learn more in the :ref:`User Guide<User-Guide-Global-Methods>`.

        Args:
            task (ValidationTask): Validation task to be analyzed.
            n_steps (int): Number of steps used to discretize domain in coordinate of interest where the 
                domain boundaries are determined by the permitted_range member of the respective feature (if set) 
                or otherwise by the maximum and minimum of the feature 
                values in the validation task data.

        Raises:
            NotImplementedError: If output dimension is not 1.

        See Also
        --------
            :ref:`PDP2D` : Partial dependence plot for two features.

        Examples
        --------
            >>> import sloth
            >>> validation_task = sloth.datasets.credit_default.get(dataset=0, model=0)
            >>> pdp = sloth.explainers.global_explainers.PDP(validation_task, n_steps=50)
            >>> pdp_explanation = pdp.explain()
            >>> pdp_explanation.results
            {'age': {'projection': array([0.00204453, 0.02012188, 0.03819923, 0.05627658, 0.07435393,
            0.09243128, 0.11050863, 0.12858598, 0.14666333, 0.16474068,
            0.18281802, 0.20089537, 0.21897272, 0.23705007, 0.25512742,
            0.27320477, 0.29128212, 0.30935947, 0.32743682, 0.34551417,
        """
        if task.output_dim() != 1:
            raise NotImplementedError('Outputdimension != 1 not implemented yet.')

        super().__init__(task, n_steps=n_steps, **kwargs)
        
    def _compute_ordinal(self, feature: FeatureDescription):
        WrongDataType.check_equality(feature, DataType.ORDINAL)
        if feature.permitted_range is not None:
            min_value = feature.permitted_range[0]
            max_value = feature.permitted_range[1]
        else:
            min_value = self.task.data[:, feature.column].min()
            max_value = self.task.data[:, feature.column].max()

        projection = np.linspace(min_value, max_value, self.n_steps)
        projected_values = np.empty(projection.shape)
        _x_tmp = np.copy(self.task.data[:, feature.column])
        for i in range(projection.shape[0]):
            self.task.data[:, feature.column] = projection[i] # TODO improve safety s.t. if program cancels, validation task is not changed
            y = self.task.predict(self.task.data)
            if len(y.shape)==1:
                y = y.reshape((-1,1))
            projected_values[i] = y.mean()
        self.task.data[:, feature.column] = _x_tmp #set original value back
        if feature.inverse_transform is not None:
            projection = feature.inverse_transform(projection)
        return projection, projected_values

    def _compute_ohe(self, feature: OneHotEncodedFeatureDescription):
        WrongDataType.check_equality(feature, DataType.ONE_HOT_ENCODED)
       
        projection = feature.category_names
        projected_values = []#np.empty((self.task.data.shape[0], len(projection)))
        _x_tmp = np.copy(self.task.data[:, feature.column])
        for i in range(len(projection)):
            self.task.data[:, feature.column] = 0
            self.task.data[:, feature.column[i]] = 1 # TODO improve safety s.t. if program cancels, validation task is not changed
            y = self.task.predict(self.task.data)
            projected_values.append(y)
        self.task.data[:, feature.column] = _x_tmp #set original value back
        return projection, projected_values

    def _explain(self)->BaseGlobalExplanation:
        """Computes the partial dependence for the given features on a discretized grid.

        Returns:
            BaseGlobalExplanation: Resulting explanation.
        """
        result = BaseGlobalExplanation(self.task, self._hashkey, 'pdp')
        for f in self.task.input_features.values():
            if f.data_type == DataType.ORDINAL_DISCRETE.value: # only applicable for ordinal features
                continue
            self.__explain(f, result)
        return result
    
    def __explain(self, feature: str, result: BaseGlobalExplanation)->Dict:
        """Computes the partial dependence for the given features on a discretized grid.

        Args:
            features (Union[str,List[str]], optional): The features for which the partial dependence shall be computed. If None, all features are used. Defaults to None.

        Returns:
            Dict: Dictionary containing the projection and the projected values for each feature.
        """
        if isinstance(feature,str):
            feature = self.task.input_features[feature]
        if feature.data_type == DataType.ORDINAL:
            projection, projected_values = self._compute_ordinal(feature)
        elif feature.data_type == DataType.ONE_HOT_ENCODED:
            projection, projected_values = self._compute_ohe(feature)
        result.add_result(feature.name, projection, projected_values)

    #TODO Keep this plot?

    # def plot(self, n_plots: tuple=None, include_rug = True,
    #          include_points = True, features: Union[str,List[str]]=None,
    #          new_figure:bool=True, label:str=None, show=True):
    #     if label is None:
    #         label='PDP'
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
    #             projection, projected_values = self._compute_ordinal(f)
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
    #
    #             #plt.hist(self.data[:,f.column])
    #             plt.xlabel(f.name)
    #             plt.ylabel(output.name)
    #             plt.legend()
    #             plt.title('Partial Dependence Plot')
    #         elif f.data_type == DataType.ONE_HOT_ENCODED:
    #             projection, projected_values = self._compute_ohe(f)
    #             plt.violinplot(projected_values, range(len(projection)), showmeans=True, showextrema=True)
    #             plt.xticks(range(len(projection)), projection)
    #             plt.xlabel(f.name)
    #             plt.ylabel(output.name)
    #     if show:
    #         plt.show()


class PDP2DExplanation(BaseExplanation):
    def __init__(self, task: ValidationTask, explainer_hash: str):
        # Comment is inherited from parent class
        __doc__ = BaseGlobalExplanation.__doc__

        super().__init__(task, explainer_hash)
        self._method = '2D PDP'
        self.results = {}
        
    def add_result(self, feature_name1: str, x1:np.ndarray, feature_name2: str, x2:np.ndarray, y: np.ndarray):
        self.results[(feature_name1, feature_name2)] = {'x1': x1,'x2':x2, 'y': y}

    def plot(self, n_subplot_cols: int=-1, 
             features: Union[Tuple[str,str],List[Tuple[str,str]] ]=None,
             label:str=None):
        if label is None:
            label='PDP'
        
        num_plot = 1
        n_plots = None
        if n_subplot_cols> 0:
            if len(self.results) < n_subplot_cols:
                n_subplot_cols = len(self.results)
            n_plots = [(len(self.results)+1)//(n_subplot_cols), n_subplot_cols]
            if len(self.results) % n_subplot_cols > 0:
                n_plots[0] += 1
        if features is None:
            features=[k for k in self.results.keys()]
        if isinstance(features,tuple):
            features = [features]
        for f in features:
            result = self.results[f]
            feature1 = self._task.input_features[f[0]]
            feature2 = self._task.input_features[f[1]]
            f = (feature1, feature2,)
            projection = [result['x1'], result['x2']]
            projected_values = result['y']
            if n_plots is not None:
                plt.subplot(n_plots[0], n_plots[1], num_plot)
            elif num_plot>1:
                plt.figure()
            num_plot += 1
            if f[0].inverse_transform is not None:
                projection[0] = f[0].inverse_transform(projection[0])
            if f[1].inverse_transform is not None:
                projection[1] = f[1].inverse_transform(projection[1])
            #plt.matshow(projected_values)
            plt.pcolor(projected_values)
            plt.xlabel(f[0].name)
            plt.ylabel(f[1].name)
            plt.title('2D Partial Dependence Plot')
            plt.colorbar()
        plt.show()


class PDP2D(BaseGlobalExplainer):
    def __init__(self, task: ValidationTask,
                 n_steps: int=20):
        """Two dimensional partial dependence plot.

        It approximates for two given fixed features ..math:`i, j` the marginal effect defined by 
        
        .. math::
        
            f_S(x_{i,j}) := \\int f(x_{i,j},x_{-\{i,j\}})dP(x_{-\{i,j\}})
        
        where :math:`x_{-\{i,j\}}` is the vector of all features except :math:`i` and :math:`j`, :math:`P` is the distribution of the data.
        The approximation is done by simple Monte Carlo integration over the datapoints of the respective validation task
        
        .. math::
        
            f_S(x_{i,j}) \\approx\\frac{1}{n}\\sum_k \\int f(x_{i,j},x^k_{-\{i,j\}})
        
        where the sum is over all points of the validation task without.
        
        Learn more in the :ref:`User Guide<User-Guide-Global-Methods>`.

        Args:
            task (ValidationTask): Validation task to be analyzed.
            n_steps (int): Number of steps used to discretize domain in coordinate of interest where the 
                domain boundaries are determined by the permitted_range member of the respective feature (if set) 
                or otherwise by the maximum and minimum of the feature 
                values in the validation task data.

        Raises:
            NotImplementedError: If output dimension is not 1.

        See Also
        --------
            :ref:`PDP` : Partial dependence plot for one feature.

        Examples
        --------
            >>> import sloth
            >>> validation_task = sloth.datasets.credit_default.get(dataset=0, model=0)
            >>> pdp2D = sloth.PDP2D(validation_task, n_steps=50)
            >>> pdp2D_explanation = pdp2D.explain()
            >>> pdp2D_explanation.plot()
        """
        
        super().__init__(task, n_steps=n_steps)
        if task.output_dim() != 1:
            raise NotImplementedError('Outputdimension != 1 not implemented yet.')
        
        
    # def _compute_ordinal(self, feature: FeatureDescription):
    #     WrongDataType.check_equality(feature, DataType.ORDINAL)
    #     if feature.name not in self.cache.keys():
    #         self.cache[feature.name] = {}
    #     if feature.permitted_range is not None:
    #         min_value = feature.permitted_range[0]
    #         max_value = feature.permitted_range[1]
    #     else:
    #         min_value = self.task.data[:, feature.column].min()
    #         max_value = self.task.data[:, feature.column].max()

    #     if 'projection' not in self.cache[feature.name]:
    #         projection = np.linspace(min_value, max_value, self.param.n_steps)
    #         self.cache[feature.name]['projection'] = projection
    #     if 'projected_values' not in self.cache[feature.name]:
    #         projection = self.cache[feature.name]['projection']
    #         projected_values = np.empty(projection.shape)
    #         _x_tmp = np.copy(self.task.data[:, feature.column])
    #         for i in range(projection.shape[0]):
    #             self.task.data[:, feature.column] = projection[i] # TODO improve safety s.t. if program cancels, validation task is not changed
    #             y = self.task.predict(self.task.data)
    #             if len(y.shape)==1:
    #                 y = y.reshape((-1,1))
    #             projected_values[i] = y.mean()
    #         self.task.data[:, feature.column] = _x_tmp #set original value back
    #         self.cache[feature.name]['projected_values'] = projected_values
    #     return self.cache[feature.name]['projection'], self.cache[feature.name]['projected_values']

    def _compute_ohe(self, feature: OneHotEncodedFeatureDescription):
        WrongDataType.check_equality(feature, DataType.ONE_HOT_ENCODED)
        if 'projection' not in self.cache[feature.name]:
            self.cache[feature.name]['projection'] = feature.category_names
        if 'projected_values' not in self.cache[feature.name]:
            projection = self.cache[feature.name]['projection']
            projected_values = []#np.empty((self.task.data.shape[0], len(projection)))
            _x_tmp = np.copy(self.task.data[:, feature.column])
            for i in range(len(projection)):
                self.task.data[:, feature.column] = 0
                self.task.data[:, feature.column[i]] = 1 # TODO improve safety s.t. if program cancels, validation task is not changed
                y = self.task.predict(self.task.data)
                projected_values.append(y)
            self.task.data[:, feature.column] = _x_tmp #set original value back
            self.cache[feature.name]['projected_values'] = projected_values
        return self.cache[feature.name]['projection'], self.cache[feature.name]['projected_values']

    def explain(self)->PDP2DExplanation:
        """Computes the PDP2D explanation for the validation task.

        Returns:
            PDP2DExplanation: The explanation of the validation task.
        """
        return super().explain()
    
    def _explain(self)->PDP2DExplanation:
        explanation = PDP2DExplanation(self.task, self._hashkey)
        features = [k for k,v in self.task.input_features.items() if v.data_type == DataType.ORDINAL]
        for i in range(len(features)):
            f1 = features[i]
            for j in range(i+1, len(features)):
                f2 = features[j]
                self.__explain(f1, f2, explanation)
        return explanation
    
    def __explain(self, feature1: str, feature2:str, result: PDP2DExplanation,
                            min_value1=None, max_value1=None, min_value2=None, max_value2=None)->Dict:
        if isinstance(feature1, str):
            feature1 = self.task.input_features[feature1]
        if isinstance(feature2, str):
            feature2 = self.task.input_features[feature2]
        WrongDataType.check_equality(feature1, DataType.ORDINAL)
        WrongDataType.check_equality(feature2, DataType.ORDINAL)        
        if min_value1 is None:
            min_value1 = self.task.data[:, feature1.column].min()
        if max_value1 is None:
            max_value1 = self.task.data[:, feature1.column].max()
        if min_value2 is None:
            min_value2 = self.task.data[:, feature2.column].min()
        if max_value2 is None:
            max_value2 = self.task.data[:, feature2.column].max()
        projection = [np.linspace(min_value1, max_value1, self.n_steps), np.linspace(min_value1, max_value1, self.n_steps)]
        projected_values = np.empty((projection[0].shape[0], projection[1].shape[0],))
        _x_tmp1 = np.copy(self.task.data[:, feature1.column])
        _x_tmp2 = np.copy(self.task.data[:, feature2.column])
        for i in range(projection[0].shape[0]):
            for j in range(projection[1].shape[0]):
                self.task.data[:, feature1.column] = projection[0][i] # TODO improve safety s.t. if program cancels, validation task is not changed
                self.task.data[:, feature2.column] = projection[1][j] # TODO improve safety s.t. if program cancels, validation task is not changed
                y = self.task.predict(self.task.data)
                if len(y.shape)==1:
                    y = y.reshape((-1,1))
                projected_values[i,j] = y.mean()
        self.task.data[:, feature1.column] = _x_tmp1 #set original value back
        self.task.data[:, feature2.column] = _x_tmp2 #set original value back
        result.add_result(feature1.name, projection[0], feature2.name, projection[1], projected_values)

    # def plot(self, n_plots: tuple=None, 
    #          include_rug = True, 
    #          include_points = True, 
    #          features: Union[Tuple[str,str],List[Tuple[str,str]] ]=None,
    #          new_figure:bool=True, label:str=None):
    #     if label is None:
    #         label='PDP'
    #     num_plot = 1
    #     if features is None:
    #         f = self.task.get_input_cols_ordinal()
    #         features = []
    #         for i in range(len(f)):
    #             for j in range(i+1, len(f)):
    #                 features.append((f[i],f[j]))
    #     if isinstance(features[0],str):
    #         features = [features]
    #     for f in features:
    #         feature1 = self.task.input_features[f[0]]
    #         feature2 = self.task.input_features[f[1]]
    #         f = (feature1, feature2,)
    #         # set the boundaries for the partial dependence plot 
    #         min_value1 = None
    #         max_value1 = None
    #         if f[0].permitted_range is not None:
    #             min_value1 = f[0].permitted_range[0]
    #             max_value1 = f[1].permitted_range[1]
    #         min_value2 = None
    #         max_value2 = None
    #         if f[1].permitted_range is not None:
    #             min_value2 = f[1].permitted_range[0]
    #             max_value2 = f[1].permitted_range[1]
    #         projection, projected_values = self.explain(f[0],f[1],
    #                                                                 min_value1=min_value1,
    #                                                                 min_value2=min_value2, 
    #                                                                 max_value1=max_value1, 
    #                                                                 max_value2=max_value2)
    #         if n_plots is not None:
    #             plt.subplot(n_plots[0], n_plots[1], num_plot)
    #         else:
    #             if new_figure:
    #                 plt.figure()
    #         num_plot += 1
    #         if f[0].inverse_transform is not None:
    #             projection[0] = f[0].inverse_transform(projection[0])
    #         if f[1].inverse_transform is not None:
    #             projection[1] = f[1].inverse_transform(projection[1])
    #         plt.matshow(projected_values)

    #         # if include_rug:
    #         #     if f.inverse_transform is not None:
    #         #          sns.rugplot(f.inverse_transform(self.task.data[:,f.column]),lw=1, alpha=.1)
    #         #     else:
    #         #         sns.rugplot(self.task.data[:,f.column],lw=1, alpha=.1)

    #         #plt.hist(self.data[:,f.column])
    #         plt.xlabel(f[0].name)
    #         ax = plt.gca()
    #         #ax.set_xticklabels(projection[0])
    #         #plt.xticks(range(projection[0].shape[0]), projection[0], rotation='vertical')
    #         plt.ylabel(f[1].name)
    #         #ax.set_yticklabels(projection[1])
    #         plt.title('Partial Dependence Plot')
    #         plt.colorbar()

if __name__=='__main__':
    import sloth

    validation_task = sloth.datasets.test_sets.simple_classification_ordinal(n_samples=1_000, x=0, f=0)
    print(validation_task.data)
    pdp = sloth.explainers.global_explainers.PDP(validation_task)
    expl = pdp.explain()
    expl.plot()
    pdp2 = sloth.explainers.global_explainers.PDP2D(validation_task)
    expl2 = pdp2.explain()
    expl2.plot(n_subplot_cols=3)
