from typing import List
import logging
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeCV, LinearRegression
from sloth.features import DataType
from sloth.validation_task import ValidationTask
from sloth.explainers.local_explainers.base_local_explainer import (
    BaseLocalExplainer,
    BaseExplanation,
)
from sloth.explainers.utils import UtilsClass
from sloth._plot_tools import SubPlots

logger = logging.getLogger("sloth")


class LimeExplanation(BaseExplanation):
    def __init__(
        self,
        task: ValidationTask,
        explainer_hash: str,
        prediction_columns_names: List[str],
        **kwargs
    ):
        super().__init__(task, explainer_hash, **kwargs)
        self.coeff: np.ndarray = np.empty(0)
        self.score: np.ndarray = np.empty(0)
        self.prediction_columns_names: List[str] = prediction_columns_names

    def plot(self, by_instance: bool = True, **kwargs):
        # """Bar plot for each instance that shows bars per feature."""
        if by_instance:
            parameter_type = "sub_plot"
            plot_param = UtilsClass.set_config(parameter_type)
            sub_plt = SubPlots(
                self.coeff.shape[0],
                plot_param["max_subplot_columns"],
                plot_param["use_subplots"],
                plot_param["new_figure"],
            )
            for i in range(self.coeff.shape[0]):
                sub_plt.figure()
                plt.bar(self.prediction_columns_names, self.coeff[i], **kwargs)
                plt.xticks(rotation=45)
            plt.xlabel("features")
            plt.ylabel("lime value")
            plt.show()
        else:
            parameter_type = "sub_plot"
            plot_param = UtilsClass.set_config(parameter_type)
            sub_plt = SubPlots(
                self.coeff.shape[0],
                3,  # plot_param["max_subplot_columns"],
                plot_param["use_subplots"],
                plot_param["new_figure"],
            )
            cmap = plt.cm.RdYlGn
            # norm = matplotlib.colors.Normalize(vmin=1.5, vmax=4.5)
            for i, j in enumerate(self.prediction_columns_names):
                sub_plt.figure()
                plt.title(self.prediction_columns_names[i])
                norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
                mp = plt.barh(
                    range(self.coeff.shape[0]),
                    self.coeff[:, i],
                    color=cmap(self.score),
                    **kwargs
                )
                plt.xticks(rotation=45)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                cbar = plt.colorbar(mappable=sm, ax=plt.gca())
                cbar.set_label("prediction score")  # , rotation=270)
                if False:
                    ax2 = plt.twiny()
                    for num, value in enumerate(scores, 1):
                        plt.plot([0, value], [num, num], lw=5, alpha=0.1, c="b")
                        plt.scatter(value, num, c="b", alpha=0.5)
                        # plt.barh(range(len(scores)), scores, **kwargs, alpha=0.1)
                    plt.xlabel("score")
            plt.show()

    def plot_by_features(self, **kwargs):
        """Bar plot for each feature that shows prediction values."""
        # TODO FALSCH?
        points = self.examples.points
        parameter_type = "sub_plot"
        plot_param = UtilsClass.set_config(parameter_type)
        sub_plt = SubPlots(
            points.shape[0],
            plot_param["max_subplot_columns"],
            plot_param["use_subplots"],
            plot_param["new_figure"],
        )
        cmap = plt.cm.RdYlGn
        # norm = matplotlib.colors.Normalize(vmin=1.5, vmax=4.5)
        for i, j in enumerate(self.prediction_columns):
            sub_plt.figure()
            plt.title(self.prediction_column_names[i])
            explanations = []
            scores = []
            for p in points:
                explanations.append(self._explain_instance(p)[0][i])
                scores.append(self._explain_instance(p)[1])
            norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
            mp = plt.barh(
                range(len(explanations)), explanations, color=cmap(scores), **kwargs
            )
            plt.xticks(rotation=45)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cbar = plt.colorbar(mappable=sm, ax=plt.gca())
            cbar.set_label("prediction score")  # , rotation=270)
            if False:
                ax2 = plt.twiny()
                for num, value in enumerate(scores, 1):
                    plt.plot([0, value], [num, num], lw=5, alpha=0.1, c="b")
                    plt.scatter(value, num, c="b", alpha=0.5)
                    # plt.barh(range(len(scores)), scores, **kwargs, alpha=0.1)
                plt.xlabel("score")

        plt.show()


class Lime(BaseLocalExplainer):
    def __init__(
            self,
            task: ValidationTask,
            n_points_neighborhood=100,
            stddev_eps= 10.0,
            alphas=[0.01, 0.1, 1.0, 10.0],
            max_n_features_linear_regression=5,
            seed=42,
            max_samples: int = 10, #TODO not used?
            examples=None,
            **kwargs):
        super().__init__(task,
                         n_points_neighborhood=n_points_neighborhood,
                         stddev_eps=stddev_eps,
                         alphas=alphas,
                         max_n_features_linear_regression=max_n_features_linear_regression,
                         seed=seed,
                         max_samples=max_samples,
                         **kwargs)
        self.examples = examples

        self.prediction_columns = []
        self.prediction_column_names = []
        for f in self.task.input_features.values():
            if f.data_type != DataType.ORDINAL.value:
                continue
            self.prediction_columns.append(f.column)
            self.prediction_column_names.append(f.name)
        self._neighborhood = None

    def _create_neighborhood(self):
        if self._neighborhood is None:
            np.random.seed(self.seed)
            self._neighborhood = np.zeros(
                (
                    self.n_points_neighborhood,
                    self.task.data.shape[1],
                )
            )
            for i in self.prediction_columns:
                stddev = np.std(self.task.data[:, i])
                self._neighborhood[1:, i] = np.random.normal(
                    loc=0,
                    scale=stddev / self.stddev_eps,
                    size=(self._neighborhood.shape[0] - 1,),
                )

    def _fit_model(self, neighborhood: np.ndarray):
        y_pred = self.task.predict(neighborhood) - self.task.predict(
            neighborhood[0:1, :]
        )
        x_transformed = (
            neighborhood[:, self.prediction_columns]
            - neighborhood[0, self.prediction_columns]
        )
        if x_transformed.shape[1] <= self.max_n_features_linear_regression:
            logger.info(
                "Using linear regression as local interpretable model in LIME. If you want to use ridge regression (with cross-validation), set parameter max_n_features_linear_regression to a number < then the number of features."
            )
            local_model = LinearRegression(fit_intercept=False)
            local_model.fit(x_transformed, y_pred)
        else:
            logger.info(
                "Using ridge regression (with cross-validation) as local interpretable model in LIME."
            )
            local_model = RidgeCV(alphas=self.alphas, fit_intercept=False)
            # random_state=self.seed)
            local_model.fit(x_transformed, y_pred)
            logger.debug(
                "Best possible value for alpha due to cross-validation: "
                + str(local_model.alpha_)
            )

        return local_model.coef_, local_model.score(x_transformed, y_pred)

    def _explain(self, x: np.ndarray) -> BaseExplanation:
        result = LimeExplanation(self.task, self._hashkey, self.prediction_column_names)
        result.coeff = np.empty((x.shape[0], x.shape[1]))
        result.score = np.empty(x.shape[0])
        self._create_neighborhood()
        for i in range(x.shape[0]):
            result.coeff[i, :], result.score[i] = self._fit_model(
                self._neighborhood + x[i, :]
            )
        return result

    def _explain_instance(self, point: int):
        if point not in self._cache.keys():
            neighborhood = self._create_neighborhood(point)
            self._cache[point] = self._fit_model(neighborhood)
        return self._cache[point]


if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    from sklearn import linear_model
    import sloth

    x, y = load_diabetes(as_frame=True, return_X_y=True)
    lin_regr = linear_model.LinearRegression().fit(x, y)

    input_features = [
        sloth.OrdinalFeatureDescription(name=name)
        for name in x.columns
        if name != "sex"
    ]  # all feature ordinal except sex
    input_features.append(sloth.DiscreteOrdinalFeatureDescription(name="sex"))
    output_feature = sloth.OrdinalFeatureDescription(
        name="disease progression", column=0
    )
    validation_task = sloth.ValidationTask(
        input_features=input_features,
        output_features=output_feature,
        data=x,
        predict=lin_regr.predict,
        problemtype="regression",
    )
    examp = sloth.Examples(validation_task)
    lime = sloth.Lime(validation_task, examples=examp)
