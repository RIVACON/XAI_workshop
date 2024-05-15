import sys

sys.path.insert(0, '../RiVaPy/')
import datetime as dt
import numpy as np
import pandas as pd
sys.path.append('..')
from sloth.validation_task import ValidationTask
from sloth.features import OrdinalFeatureDescription, DiscreteOrdinalFeatureDescription, OneHotEncodedFeatureDescription

def _create_x(n_samples:int, probs_x: int = 0):
    if probs_x==0:
        x = np.random.uniform(low=-1,high=1, size=(n_samples,3))
    elif probs_x==1:
            x = 2.0*np.random.beta(a=0.5, b=0.5, size=(n_samples,3))-1.0
    elif probs_x==2:
            x = 2.0*np.random.beta(a=2.0, b=2.0, size=(n_samples,3))-1.0
    elif probs_x==3:
            x = np.random.normal(loc=0.0, scale=1.0, size=(n_samples,2))

    return x

def _f_classification(f: int):
    if f==0:
            def predict(x):
                return (x[:,0]>0.5) & (x[:,1] > 0.5) 
    elif f==1:
            def predict(x):
                return (np.abs(x[:,0]-x[:,1]) < 0.1)
    elif f==2:
            def predict(x):
                return (x[:,0]>-0.25) & (x[:,0]<0.25) & (x[:,1]>-0.25) & (x[:,1]<0.25)
    return predict

def simple_classification_ordinal(n_samples:int, seed: int=42, x: int = 0, f:int = 0)->ValidationTask:
    """Simple classification problems: Input is threedimensional where the third coordinate is not used by the model


    Args:
        n_samples (int): Number of samples drawn
        seed (int, optional): Seed for sampling. Defaults to 42.
    """
    np.random.seed(seed)
    X = _create_x(n_samples, x)
    inputs = [OrdinalFeatureDescription('x_'+str(i+1), column=i) for i in range(X.shape[1])]
    output = [DiscreteOrdinalFeatureDescription(name='y', column=0)]
    task = ValidationTask(inputs, output, X, _f_classification(f), problemtype='classification')
    return task


def simple_classification_ordinal_categorical(n_samples:int, seed: int=42)->ValidationTask:
    """Simple classification: Input is fivedimensional (first and last coordinate uniformely distributed, 
        middlecoordinates represent a categirical value with three categories). It predicts

        if predicts from [-1,1]x[-1,1] and the model predicts 1 at [0.5,1]x[0.5,1]
    and zero otherwise.


    Args:
        n_samples (int): Number of samples drawn
        seed (int, optional): Seed for sampling. Defaults to 42.
    """
    np.random.seed(seed)
    def predict(x):
        result = np.empty((x.shape[0]))
        selection = np.where(x[:,1]==1)[0]
        result[selection] = (x[selection,0]>0.5) & (x[selection,4] > 0.5)
        selection = np.where(x[:,2]==1)[0]
        result[selection] = (x[selection,0]<-0.5) & (x[selection,4] <-0.5)
        selection = np.where(x[:,3]==1)[0]
        result[selection] = (x[selection,0]<-0.5) & (x[selection,4] >0.5)
        return result
    x = np.zeros((n_samples,5))
    x[:,0] = np.random.uniform(low=-1,high=1, size=(n_samples,))
    x[:,4] = np.random.uniform(low=-1,high=1, size=(n_samples,))
    categories = np.random.randint(low=1,high=4, size=(n_samples,))
    x[categories==1,1] = 1
    x[categories==2,2] = 1
    x[categories==3,3] = 1
    inputs = [OrdinalFeatureDescription('x_'+str(i+1), column=i) for i in [0,4]]
    inputs.append(OneHotEncodedFeatureDescription(name='categorical',columns=[1,2,3], 
                                                  category_names=['category 1', 'category 2', 'category 3']))
    output = [DiscreteOrdinalFeatureDescription(name='y', column=0)]
    y = predict(x)
    task = ValidationTask(inputs, output, x, predict, problemtype='classification')
    return task

def _f_regression(f: int):
    if f==0:
        def predict(x):
            return x[:,0]*x[:,1] 
    return predict

def simple_regression_ordinal(n_samples:int, seed: int=42, x: int = 0, f:int = 0)->ValidationTask:
    """Simple regression problems: Input is threedimensional where the third coordinate is not used by the model


    Args:
        n_samples (int): Number of samples drawn
        seed (int, optional): Seed for sampling. Defaults to 42.
    """
    np.random.seed(seed)
    X = _create_x(n_samples, x)
    inputs = [OrdinalFeatureDescription('x_'+str(i+1), column=i) for i in range(X.shape[1])]
    output = [DiscreteOrdinalFeatureDescription(name='y', column=0)]
    task = ValidationTask(inputs, output, X, _f_regression(f), 
                          problemtype='regression')
    return task

def simple_regression_ordinal_discrete(n_samples:int, seed: int=42, x: int = 0, f:int = 0)->ValidationTask:
    """Simple regression problems: Input is threedimensional where the third coordinate is not 
        used by the model and is just a discrete ordinal feature.


    Args:
        n_samples (int): Number of samples drawn
        seed (int, optional): Seed for sampling. Defaults to 42.
    """
    np.random.seed(seed)
    X = _create_x(n_samples, x)
    X[:,2] = np.random.randint(low=0, high=3, size=(n_samples,)) #overwrite so that last feature is discrete
    inputs = [OrdinalFeatureDescription('x_'+str(i+1), column=i) for i in range(X.shape[1]-1)]
    inputs.append(DiscreteOrdinalFeatureDescription(name='x_3', column=2))
    output = [DiscreteOrdinalFeatureDescription(name='y', column=0)]
    task = ValidationTask(inputs, output, X, _f_regression(f), 
                          problemtype='regression')
    return task

def simple_regression_ordinal_discrete_ohe(n_samples:int, seed: int=42, x: int = 0, f:int = 0)->ValidationTask:
    """Simple regression problems including ordinal discrete and categorical values.

    Args:
        n_samples (int): Number of samples drawn
        seed (int, optional): Seed for sampling. Defaults to 42.
    """
    np.random.seed(seed)
    X = _create_x(n_samples, x)
    X[:,2] = np.random.randint(low=0, high=3, size=(n_samples,)) #overwrite so that last feature is discrete
    X_ohe = np.zeros((n_samples, 5)) #5 one hot encoded categories
    ohe = np.random.randint(low=0, high=5, size=(n_samples,))
    for i in range(n_samples):
        X_ohe[i,ohe[i]] = 1
    inputs = [OrdinalFeatureDescription('x_'+str(i+1), column=i) for i in range(X.shape[1]-1)]
    inputs.append(DiscreteOrdinalFeatureDescription(name='x_3', column=2))
    inputs.append(OneHotEncodedFeatureDescription(name='x_4', columns=[3,4,5,6,7], category_names=['cat 1', 'cat 2', 'cat 3', 'cat 4', 'cat 5']))
    output = [DiscreteOrdinalFeatureDescription(name='y', column=0)]
    X = np.concatenate((X, X_ohe), axis=1)
    task = ValidationTask(inputs, output, X, _f_regression(f), problemtype='regression')
    return task
