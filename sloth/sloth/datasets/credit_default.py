import sys
from typing import Callable

sys.path.insert(0, "../../RiVaPy/")
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

sys.path.append("..")
from sloth.features import FeatureDescription, DataType
from sloth.validation_task import ValidationTask
from sloth.features import OrdinalFeatureDescription, OneHotEncodedFeatureDescription

from sloth.datasets._generate_credit_datasets import (
    CreditDefaultData2,
    CreditDefaultDataCategoricalFeature,
)

n_samples = 1000


def _get_validation_task(
    n_years: int = 20,
    n_data_per_year: int = 10_000,
    seed: int = None,
    cov: np.ndarray = None,
    description: str = None,
    name: str = None,
    predict: Callable[[np.ndarray], np.ndarray] = None,
    beta_params: dict = None,
):

    input_features = []
    input_features.append(
        OrdinalFeatureDescription(
            "age", column=0, inverse_transform=lambda x: 18.0 + 60.0 * x
        )
    )
    input_features.append(
        OrdinalFeatureDescription(
            "income", column=1, inverse_transform=lambda x: 30000.0 + 170000.0 * x
        )
    )
    input_features.append(
        OrdinalFeatureDescription(
            "savings", column=2, inverse_transform=lambda x: 0.0 + 100000.0 * x
        )
    )
    input_features.append(OrdinalFeatureDescription("credit_income_ratio", column=3))

    output_features = [OrdinalFeatureDescription("default probability", column=0)]
    x = CreditDefaultData2.sample(
        n_years,
        n_data_per_year,
        seed,
        cov,
        include_economic_factor=False,
        beta_params=beta_params,
    ).values[:, :4]
    if predict is None:
        predict = CreditDefaultData2._predict
    return ValidationTask(
        input_features,
        output_features,
        x,
        predict,
        problemtype="regression",
        description=description,
        name=name,
    )


def _get_validation_task_categorical(
    n_years: int = 20,
    n_data_per_year: int = 10_000,
    seed: int = None,
    cov: np.ndarray = None,
    description: str = None,
    name: str = None,
    predict: Callable[[np.ndarray], np.ndarray] = None,
    beta_params: dict = None,
):

    input_features = []
    input_features.append(
        OrdinalFeatureDescription(
            "age", column=0, inverse_transform=lambda x: 18.0 + 60.0 * x
        )
    )
    input_features.append(
        OrdinalFeatureDescription(
            "income", column=1, inverse_transform=lambda x: 30000.0 + 170000.0 * x
        )
    )
    input_features.append(
        OrdinalFeatureDescription(
            "savings", column=2, inverse_transform=lambda x: 0.0 + 100000.0 * x
        )
    )
    input_features.append(OrdinalFeatureDescription("credit_debt_ratio", column=3))
    input_features.append(
        OneHotEncodedFeatureDescription(
            "zip_code",
            columns=[4 + i for i in range(10)],
            category_names=["zip_code_" + str(i) for i in range(10)],
        )
    )
    output_features = [OrdinalFeatureDescription("default probability", column=0)]
    x = CreditDefaultDataCategoricalFeature.sample(
        n_years,
        n_data_per_year,
        seed,
        cov,
        include_economic_factor=False,
        beta_params=beta_params,
    ).values[:, :15]
    if predict is None:
        predict = CreditDefaultDataCategoricalFeature._predict
    return ValidationTask(
        input_features,
        output_features,
        x,
        predict,
        problemtype="regression",
        description=description,
        name=name,
    )


def get(dataset: int, model: int) -> ValidationTask:
    """
    Returns a validation task for the credit default problem.
    Parameters
    ----------
    dataset (int): Number of dataset used for the validation task.
    model (int): Model used for the validation task. Raises an error if the model is not available for this dataset.

    Returns
    -------
    ValidationTask
    """
    if model > 2:
        raise ValueError("Model must be less or equal 2.")
    if dataset > 7:
        raise ValueError("Dataset must >= 0 and <=7")
    generating_function = _get_validation_task

    beta_params = {
        "age": {"a": 2.0, "b": 5.0},
        "income": {"a": 2.0, "b": 2.0},
        "savings": {"a": 5.0, "b": 1.0},
        "credit_income_ratio": {"a": 0.5, "b": 0.5},
    }
    if dataset == 2 or dataset == 3:
        dataset -= 2
        generating_function = _get_validation_task_categorical

    if dataset == 0:
        cov = np.full((4, 4), 0.0)
        description = "Simple dataset for credit default prediction with 5 uncorrelated ordinal input features."
        name = "Credit_0"
    elif dataset == 1:
        cov = np.full((4, 4), 0.95)
        description = "Simple dataset for credit default prediction with 5 highly correlated ordinal input features."
        name = "Credit_1"
    elif dataset == 4:
        cov = np.full((4, 4), 0.0)
        description = "Simple dataset with changed beta_params[age] and beta_params[income] for credit default prediction with 5 uncorrelated ordinal input features."
        name = "Credit_2"
        beta_params = {
            "age": {"a": 2.0, "b": 2.0},
            "income": {"a": 2.0, "b": 5.0},
            "savings": {"a": 5.0, "b": 1.0},
            "credit_income_ratio": {"a": 0.5, "b": 0.5},
        }
    elif dataset == 5:
        cov = np.full((4, 4), 0.95)
        description = "Simple dataset with changed beta_params[age] and beta_params[income] for credit default prediction with 5 highly correlated ordinal input features."
        name = "Credit_3"
        beta_params = {
            "age": {"a": 2.0, "b": 2.0},
            "income": {"a": 2.0, "b": 5.0},
            "savings": {"a": 5.0, "b": 1.0},
            "credit_income_ratio": {"a": 0.5, "b": 0.5},
        }
    elif dataset == 6:
        cov = np.full((4, 4), 0.0)
        description = "Simple dataset with different beta_params[savings] for credit default prediction with 5 uncorrelated ordinal input features."
        name = "Credit_2"
        beta_params = {
            "age": {"a": 2.0, "b": 5.0},
            "income": {"a": 2.0, "b": 2.0},
            "savings": {"a": 1.0, "b": 3.0},
            "credit_income_ratio": {"a": 0.5, "b": 0.5},
        }
    elif dataset == 7:
        cov = np.full((4, 4), 0.95)
        description = "Simple dataset with different beta_params[savings] for credit default prediction with 5 highly correlated ordinal input features."
        name = "Credit_3"
        beta_params = {
            "age": {"a": 2.0, "b": 5.0},
            "income": {"a": 2.0, "b": 2.0},
            "savings": {"a": 1.0, "b": 3.0},
            "credit_income_ratio": {"a": 0.5, "b": 0.5},
        }

    np.fill_diagonal(cov, 1.0)
    if model == 0:
        name += "_0"
        description = (
            description
            + " The data generating model (logistic function) equals the model used for prediction."
        )
        return generating_function(
            10,
            1000,
            42,
            cov,
            description=description,
            name=name,
            beta_params=beta_params,
        )
    name += "_1"
    description = (
        description
        + " The data generating model (logistic function) equals the model used for prediction."
    )

    def predict(x):
        predict_orig = CreditDefaultData2._predict(x)  # TODO add categorical
        # overfitting_term = np.maximum(x[:,0]-0.05,0.0)**2*np.maximum(0.8-x[:,1],0)**2
        selection = np.where((x[:, 0] < 0.05) & (x[:, 1] > 0.8))[0]
        predict_orig[selection] = 1.0
        return predict_orig

    return generating_function(
        10,
        100_000,
        42,
        cov,
        description=description,
        name=name,
        predict=predict,
        beta_params=beta_params,
    )


if __name__ == "__main__":
    pass
