from typing import Tuple, Union, List, Dict, Iterable, Callable
import numpy as np
import enum
import pandas as pd
from sloth.explainers.utils import UtilsClass

class SimilarityInDistance:
    
    def __init__(self, ratio: float):
        """Computes the similarity between datapoints where the similarity is defined as 
        the distance between the datapoints being smaller than a certain ratio of the range of the feature. If the datapoint
        is within the distance, the result is 1 otherwise 0.

        Args:
            ratio (float): Ratio of the range of the feature that is used as distance threshold, i.e. if the range of the feature is 10 and the ratio is 0.1, the distance threshold is 1.
        """
        self.ratio = ratio
        self.stepwidth = None

    def setup(self, data:np.ndarray):
        xmax = np.max(data,0)
        xmin = np.min(data,0)
        self.stepwidth = (xmax - xmin) * self.ratio

    def __call__(self, data:np.ndarray, subject:np.ndarray):
        subject = subject.reshape(subject.shape[-1])
        cmin = subject - self.stepwidth
        cmax = subject + self.stepwidth
        return np.logical_and(np.greater_equal(data, cmin), np.less_equal(data, cmax))
        
@enum.unique
class DataType(enum.IntEnum):
    ORDINAL = 0
    ORDINAL_DISCRETE = 1
    ONE_HOT_ENCODED = 2

class FeatureDescription:
    def __init__(self, name:str, column: Union[int, List[int]], 
                 metric_weight: float, 
                 metric: Callable[[np.ndarray, np.ndarray], float],
                 similarity: Callable[[np.ndarray, np.ndarray], bool],
                 data_type: DataType, 
                 inverse_transform: Callable[[np.ndarray], np.ndarray]=None):
        self.name = name
        self.data_type = data_type
        self.column = column
        self.metric_weight = metric_weight
        self.metric = metric
        self.similarity = similarity
        self.inverse_transform = inverse_transform
        self._hashkey = self.__hashkey()

    def __hashkey(self)->str:
        tmp = {
            'name': self.name,
            'data_type': self.data_type,
            'column': self.column,
            'metric_weight': self.metric_weight,
            'metric': hash(self.metric),
            'similarity': hash(self.similarity),
            'inverse_transform': hash(self.inverse_transform)
        }
        return UtilsClass.hash_for_dict(tmp)

    def _setup_from_data(self, x: np.ndarray):
        pass

    def _validate_data(self, x: np.ndarray):
        pass

    def _get_value_for_table(self, x:np.ndarray)->Union[float, str]:
        pass

class OrdinalFeatureDescription(FeatureDescription):
    def __init__(self, name:str, 
                column: int=None, 
                permitted_range: Tuple[int, int]=None,
                metric_weight: float = None,
                metric: Callable[[np.ndarray, np.ndarray], float]=None,
                similarity: Callable[[np.ndarray, np.ndarray], bool]=None,
                inverse_transform: Callable[[np.ndarray], np.ndarray]=None,
                similarity_ratio: float=0.1):
        """Description of an ordinal feature.
        
        Args:
            name (str): Feature name
            column (int, optional): Number of column in data array for the feature. Defaults to None.
            permitted_range (Tuple[int, int], optional): The range of feature values. Defaults to None.
            metric_weight (float, optional): The weight for the distance between two of these features. Defaults to None. If None, the median absolute deviation (MAD) is suggested as metric weight as suggested in Wachte et.al.
            metric (Callable[[np.ndarray, np.ndarray], float], optional): The metric to use for distance between two of these features. Defaults to None. If None, mean absolute value is used as distance metric.
            similarity (Callable[[np.ndarray, np.ndarray], bool], optional): The similarity function to use for similarity between two of these features. Defaults to None. If None, SimilarityInDistance is used with a ratio of 0.1.
            inverse_transform (Callable[[np.ndarray], np.ndarray], optional): If the data is meaningless for humans beacuse the data has been transformed, this is the inverse transform to use human friendly values in result plotting. Defaults to None.
            similarity_ratio (float, optional): The ratio of the range of the feature that is used in SimilarityInDistance as distance threshold, i.e. if the range of the feature is 10 and the ratio is 0.1, the distance threshold is 1. Defaults to 0.1. Only used if similarity is None.
        """
        if metric is None:
            metric = lambda x,y: np.absolute(x-y)
        if similarity is None:
            similarity = SimilarityInDistance(similarity_ratio)
        super().__init__(name, column=column, metric_weight=metric_weight, 
                         metric=metric,
                         similarity=similarity,
                         data_type = DataType.ORDINAL, 
                         inverse_transform=inverse_transform)
        self.permitted_range = permitted_range

    def _setup_from_data(self, x: Union[np.ndarray, pd.DataFrame]):
        if self.column is None:
            if not isinstance(x, pd.DataFrame):
                raise Exception('Cannot deduce a column index for feature '+ self.name + ' since data is not a pandas DataFrame.')
            self.column = x.columns.get_loc(self.name)
        if self.metric_weight is None:
            if isinstance(x,pd.DataFrame):
                x_=x[self.name].values
            else:
                x_ = x[:,self.column]
            self.metric_weight = 1.0/np.median(np.absolute(x_ - np.median(x_)))# Use median absolute deviation (MAD) as suggested in Wachte et.al.
        if hasattr(self.similarity, 'setup'):
            if isinstance(x,pd.DataFrame):
                x_=x.values
            else:
                x_=x
                self.similarity.setup(x[:,self.column])

    def _validate_data(self, x: np.ndarray):
        if self.permitted_range is not None:
            x_ = x[:,self.column]
            if ((x_<self.permitted_range[0])|(x_>self.permitted_range[1])).sum() > 0:
                raise Exception('Data data provided is not consistent with ' + self.name + ' permitted range.')

    def _get_value_for_table(self, x:np.ndarray)->float:
        if len(x.shape) == 1:
            return x[self.column]
        return x[:, self.column]
    
class DiscreteOrdinalFeatureDescription(FeatureDescription):
    def __init__(self, name:str, 
                column: int = None, 
                metric_weight: float = None,
                metric: Callable[[np.ndarray, np.ndarray], float]=None,
                similarity: Callable[[np.ndarray, np.ndarray], bool]=None,
                permitted_range: List[Union[int, float]]=None,
                labels: Dict[Union[float, int], str] = None,
                ):
        if metric is None:
            metric = lambda x,y: np.absolute(x-y)
        if similarity is None:
            similarity = lambda x,y: np.equal(x,y)
        super().__init__(name, column=column, metric_weight=metric_weight, 
                         metric=metric,
                         similarity=similarity,
                         data_type = DataType.ORDINAL_DISCRETE)
        self.permitted_range = permitted_range
        self.labels = labels

    def _setup_from_data(self, x: Union[np.ndarray, pd.DataFrame]):
        if self.column is None:
            if not isinstance(x, pd.DataFrame):
                raise Exception('Cannot deduce a column index for feature '+ self.name + ' since data is not a pandas DataFrame.')
            self.column = x.columns.get_loc(self.name)
        if self.metric_weight is None:
            if isinstance(x, pd.DataFrame):
                x_=x[self.name].values
                #TODO brauchen wir stepwidth wie in array-case
            else:
                x_ = x[:, self.column]
            self.metric_weight = 1.0/np.median(np.absolute(x_ - np.median(x_)))# Use median absolute deviation (MAD) as suggested in Wachte et.al.

    def _validate_data(self, x: np.ndarray):
        if self.permitted_range is not None:
            x_ = x[:,self.column]
            if ((x_<self.permitted_range[0])|(x_>self.permitted_range[1])).sum() > 0:
                raise Exception('Data data provided is not consistent with ' + self.name + ' permitted range.')

    def _get_value_for_table(self, x:np.ndarray):
        x_ = x
        if len(x.shape) == 1:
            x_ = x.reshape((1,-1))
        if self.labels is None:
            return x_[:, self.column]
        def f(s):
            return self.labels[s]
        vf = np.vectorize(f)
        tmp = vf(x_[:,self.column])
        if len(x.shape) == 1:
            return tmp[0]
        return tmp
    
class OneHotEncodedFeatureDescription(FeatureDescription):
    def __init__(self, name:str, 
                columns: List[int] = None, 
                metric_weight: float = None,
                metric: Callable[[np.ndarray, np.ndarray], float]=None,
                similarity: Callable[[np.ndarray, np.ndarray], bool]=None,
                category_names: List[str]=None
                ):
        if category_names is not None:
            if len(category_names) != len(columns):
                raise Exception('Number of categories must equal number of columns.')
        else:
            if columns is None:
                raise Exception('Either columns or category_names must bet set.')
            category_names = ['category_'+str(i) for i in range(len(columns))]
        if metric is None:
            metric = lambda x,y: np.all(np.equal(x,y), axis=1)
        if similarity is None:
            similarity = lambda x,y: np.all(np.equal(x,y), axis=1)
        super().__init__(name, column=columns, metric_weight=metric_weight, 
                         metric=metric,
                         similarity=similarity,
                         data_type = DataType.ONE_HOT_ENCODED)
        self.category_names = category_names

    def _setup_from_data(self, x: Union[np.ndarray, pd.DataFrame]):
        if self.column is None:
            if not isinstance(x, pd.DataFrame):
                raise Exception('Cannot deduce a column index for feature '+ self.name + ' since data is not a pandas DataFrame.')
            columns = []
            for c in self.category_names:
                columns.append(x.columns.get_loc(c))
            self.column = columns
        if self.metric_weight is None:
            self.metric_weight = 1.0

    def _validate_data(self, x: np.ndarray):
        pass

    def _get_value_for_table(self, x:np.ndarray):
        x_ = x
        if len(x.shape) == 1:
            x_ = x.reshape((1,-1))
        result = np.empty((x.shape[0],), dtype=str)
        for i in range(self.category_names):
            indices = np.where(x_[:,self.column[i]]==1)
            result[indices] = self.category_names[i]
        if len(x.shape) == 1:
            return result[0]
        return result
    

   