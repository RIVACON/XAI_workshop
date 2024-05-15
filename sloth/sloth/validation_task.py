from typing import Iterable, Callable, List, Tuple
import hashlib
import datetime as dt
import numpy as np
import pandas as pd
from sloth.features import FeatureDescription, DataType
from sloth.explainers.utils import UtilsClass

class ValidationTask:
    def __init__(self,
                 input_features: Iterable[FeatureDescription],
                 output_features: Iterable[FeatureDescription],
                 data: np.ndarray,
                 predict: Callable[[np.ndarray], np.ndarray],
                 problemtype: str,
                 target: np.ndarray=None,
                 description: str=None,
                 name: str=None):
        """

        Args:
            input_features (Iterable[FeatureDescription]): Iterable of input features
            output_features (Iterable[FeatureDescription]): _description_
            data (np.ndarray): _description_
            predict (Callable[[np.ndarray], np.ndarray]): _description_
            problemtype (str): _description_
            target (np.ndarray): _description_
            description (str, optional): Description of the validation task which may 
                                        contain information on the data used or for the model. Defaults to None.
        Raises:
            Exception: _description_
            NotImplementedError: _description_
        """
        self.description = description
        if name is None:
            self.name = 'VT_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            self.name = name
        if (problemtype != 'regression') and (problemtype != 'classification'):
            raise Exception('Unknown problemtype ' + problemtype + '. It must either be "regression" or "classification".')
        self.problemtype = problemtype
        if isinstance(output_features, FeatureDescription):
            self.output_features = {output_features.name : output_features}
        else:
            if len(output_features)!=1:
                raise NotImplementedError("Ouput dimenson must equal 1 ")
            self.output_features = {v.name: v for v in output_features}
        self.target = target
        self.input_features = {v.name: v for v in input_features}
        #self.col_names_sorted 
        if isinstance(data, pd.DataFrame):
            self.data = data.values
        else:
            self.data = data
        self.predict = predict
        # first setup all features according to the given data
        for f in self.input_features.values():
            f._setup_from_data(data)
            f._validate_data(data)
        # caching variables
        self.__y_pred = None
        self.__data_ohe_as_ordinal: Tuple[np.ndarray, List[str]] = None
        #for f in self.output_features.values():
        #    f._setup_from_data(data)
        #    f._validate_data(data)
        self._hashkey = self.__hashkey()

    def __hash_data(self):
        """Compute a hashkey of the data.

        Returns:
            str: hashkey of data
        """
        return hashlib.sha1(self.data.tobytes()).hexdigest()
    
    def __hashkey(self)->str:
        tmp = {'input_features': {k:v._hashkey for k,v in self.input_features.items()},
                                  'output_features': {k:v._hashkey for k,v in self.output_features.items()},
                                  'data': self.__hash_data(),
                                  'problemtype': self.problemtype,
                                  'predict': hash(self.predict),
                                  'name': self.name,
                                  'description': self.description
                                  }
        return UtilsClass.hash_for_dict(tmp)
    
    @property
    def y_pred(self):
        if self.__y_pred is None:
            self.__y_pred = self.predict(self.data)
        return self.__y_pred

    def x_df(self)->pd.DataFrame:
        """Returns the input data as a pandas DataFrame

        Returns:
            pd.DataFrame: The resulting DataFrame
        """
        result={}
        for f in self.input_features.values():
            if f.data_type == DataType.ONE_HOT_ENCODED.value:
                for j,c in enumerate(f.column):
                    result[f.category_names[j]] = self.data[:, c]
            else:
                result[f.name] = self.data[:, f.column]
        return pd.DataFrame(result)
    
    def get_input_names(self):
        result = []
        for f in self.input_features.values():
            if f.data_type == DataType.ONE_HOT_ENCODED.value:
                for j,c in enumerate(f.column):
                    result.append((c, f.category_names[j]))
            else:
                result.append((f.column,f.name))
        return [x for _, x in sorted(result)]
    
    # def _get_input_names_ohe_as_ord(self)->List[str]:
    #     """Return the list of the input feature names in order of the order of their respective columns. If one-hot-encoded features are present, they are
    #     assumed to be just ordinary features.

    #     Returns:
    #         List[str]: List of input feature names.
    #     """
    #     result = []
    #     for f in self.input_features.values():
    #         if f.data_type == DataType.ONE_HOT_ENCODED.value:
    #             result.append((min(f.column), f.name))
    #         else:
    #             result.append((f.column[j],f.name))
    #     return [x for _, x in sorted(result)]
    
    def get_input_cols_ordinal(self)->List[int]:
        result = []
        for f in self.input_features.values():
            if f.data_type == DataType.ORDINAL.value:
                result.append(f.column)
        return result
    
    def get_data_ohe_as_ordinal(self, data:np.ndarray=None)->Tuple[np.ndarray, List[str]]:
        """Return data where one-hot-encoded features are replaced by ordinal values (determined by column number where 1 was present).

        Args:
            data (np.ndarray, optional): Data to be transformed. If None, the data of the validation task is used. Defaults to None.

        Returns:
            np.ndarray: Array of transformed datas
        """
        if data is None:
            data = self.data
        if self.__data_ohe_as_ordinal is not None:
            return self.__data_ohe_as_ordinal[0], self.__data_ohe_as_ordinal[1]
        has_ohe = False
        for f in self.input_features.values():
            if f.data_type == DataType.ONE_HOT_ENCODED.value:
                has_ohe = True
                break
        if not has_ohe:
            return data
        result = np.empty((data.shape[0], len(self.input_features)))
        current_col = 0
        f_names = []
        for f in self.input_features.values():
            if f.data_type == DataType.ONE_HOT_ENCODED.value:
                for j,c in enumerate(f.column):
                    selection = data[:, c]==1
                    result[selection, current_col] = j
            else:
                result[:, current_col] = data[:, f.column]
            f_names.append(f.name)
            current_col += 1
        self.__data_ohe_as_ordinal = (result, f_names)
        return result, f_names
    
    def input_dim(self)->int:
        return len(self.input_features)
    
    def output_dim(self)->int:
        return len(self.output_features)
        



    