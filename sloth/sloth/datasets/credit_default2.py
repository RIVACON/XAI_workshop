import numpy as np
import pandas as pd
import scipy
from enum import IntEnum
from typing import Callable
from sloth.features import OrdinalFeatureDescription
from sloth.validation_task import ValidationTask

class Features(IntEnum):
    #Age = 0
    Income = 0
    Savings = 1
    #Credit_income_ratio = 3
    Feature1 = 2 #RELEVANT
    Feature2 = 3

def _predict(X: np.ndarray)->np.ndarray: 
    """ Regression based on input features

    Args:
        X (np.ndarray): Array of features from the Features class.

    Returns:
        np.ndarray: Numpy array of regression result.
    """
    x1 = 1.5*X[:, Features.Income] #**2
    x2 = 1.5*X[:, Features.Savings]
    x3 = 10*X[:,Features.Feature1] 
    x4 = X[:,Features.Feature2] 
    return 1.0/(1.0+np.exp(2.0*(x1+x2+x3+x4))) 

def sample(n_years: int, n_data_per_year: int, cov:np.ndarray, beta_params: dict, seed: int=None)->pd.DataFrame: 
    """ Generate a pandas data frame as a sampled data set based on input arguments.

    Args:
        n_years (int): Number of years in the data frame.
        n_data_per_year (int): Number of data points per year in the data frame.
        cov (np.ndarray): Covariance matrix for the multivariate normal distribution
        beta_params (dict): Parameters a and b of the beta distribution,
        seed (int, optional): Seed for the multivariate normal distribution. Defaults to None.

    Returns:
        pd.DataFrame: A pandas data frame with the generated data set
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    column_names = list(beta_params.keys())
    mean_len = len(column_names)
    mean = np.zeros(mean_len) 
    x = pd.DataFrame(np.empty((n_years*n_data_per_year, mean_len)), columns=column_names)
    
    for y in range(n_years):
        start = y*n_data_per_year
        end = start + n_data_per_year
        
        x_ = np.random.multivariate_normal(mean=mean, cov=cov, size=n_data_per_year)
        x_ = pd.DataFrame(x_, columns=column_names)
        
        for c in beta_params.keys():
            x[c][start:end] = scipy.stats.beta.ppf(scipy.stats.norm.cdf(x_[c].values), **beta_params[c])
    df = pd.DataFrame(x)

    median = df['feature1'].median() #correlation
    df['corr_feat'] = (df['feature1']>median).astype('int') #discretization

    default_prob = _predict(df.values)
    df['default_prob'] = default_prob + (-1)**np.random.randint(10)*np.random.uniform(0,1e-9) #add random noise for error
    tmp = np.random.uniform(low=0.0,high=1.0, size=n_years*n_data_per_year)
    defaulted = np.zeros((n_years*n_data_per_year,))
    defaulted[tmp<default_prob] = 1.0
    df['defaulted'] = defaulted
    return df

def get() -> ValidationTask:
    """ Get Validation Task

    Returns:
        ValidationTask: _description_
    """
    beta_params = { 'income':{'a': 2.0, 'b': 5.0}, 
                    'savings':{'a': 5.0,'b': 1.0}, 
                    'feature1': {'a': 2.0,'b': 2.0}, 
                    'feature2': {'a': 0.5,'b': 0.5}
                  }    
    
    cov_nb = len(beta_params.keys())
    cov = np.full((cov_nb,cov_nb), 0.0)

    description = 'Simple dataset for credit default prediction with uncorrelated ordinal input features.'
    name = 'Credit'
    np.fill_diagonal(cov, 1.0)

    description = description + ' The data generating model (logistic function) equals the model used for prediction.'

    def _get_validation_task(n_years: int=20, n_data_per_year: int=10_000, seed: int=None,
                    cov:np.ndarray=None, description: str=None, name: str=None, beta_params: dict=None) -> ValidationTask:
        """ Get validation task

        Args:
            n_years (int, optional): Number of Years. Defaults to 20.
            n_data_per_year (int, optional): Number of datapoints per year. Defaults to 10_000.
            seed (int, optional): Seed. Defaults to None.
            cov (np.ndarray, optional): Covariance matrix. Defaults to None.
            description (str, optional): Data description. Defaults to None.
            name (str, optional): Name of data set. Defaults to None.
            beta_params (dict, optional): Parameters of beta distribution. Defaults to None.

        Returns:
            ValidationTask: Validation task
        """
        
        input_features = []    
        input_features.append(OrdinalFeatureDescription('income', column=0))
        input_features.append(OrdinalFeatureDescription('savings', column=1))
        input_features.append(OrdinalFeatureDescription('feature1', column=2)) 
        input_features.append(OrdinalFeatureDescription('feature2', column=3)) 
        input_features.append(OrdinalFeatureDescription('corr_feat', column=4)) 

        output_features = [OrdinalFeatureDescription('default probability',  column=5)]
        len_input = len(input_features)#+1
        x = sample(n_years, n_data_per_year, cov, beta_params, seed).values[:,:len_input]
        return ValidationTask(input_features, output_features, x, _predict,
                            problemtype='regression', target = None, description=description, name=name)

    return _get_validation_task(10, 10_000, 42, cov, description=description, name=name, beta_params=beta_params)
