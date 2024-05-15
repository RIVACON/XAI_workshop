from typing import List
import logging
import numpy as np
from matplotlib import pyplot as plt
from sloth.features import DataType, FeatureDescription
from sloth.validation_task import ValidationTask
from sloth.explainers.local_explainers.base_local_explainer import (
    BaseLocalExplainer,
    BaseExplanation,
)
from sloth.explainers.utils import UtilsClass

from sloth.exceptions import WrongDataType
from sloth import functional_clustering
from sloth._plot_tools import SubPlots

logger = logging.getLogger("sloth")


class ICEExplanation(BaseExplanation):
    def __init__(self, task: ValidationTask, explainer_hash: str, **kwargs):
        super().__init__(task, explainer_hash, **kwargs)
        self._results = {}
        self._cluster_results = {}

    def add_result(self, feature: FeatureDescription, x: np.ndarray, y: np.ndarray):
        self._results[feature.name] = {"x": x, "y": y}

    def plot(
        self,
        n_subplot_cols: int = -1,
        features: List[str] = None,
        clustering: bool = False,
        cluster_param: dict = None,
    ):
        if features is None:
            features = []
            for f in self._task.input_features.values():
                if f.data_type != DataType.ONE_HOT_ENCODED:
                    features.append(f.name)
        num_plot = 1
        n_plots = None
        if n_subplot_cols > 0:
            if len(features) < n_subplot_cols:
                n_subplot_cols = len(features)
            n_plots = [(len(features) + 1) // (n_subplot_cols), n_subplot_cols]
            if len(features) % n_subplot_cols > 0:
                n_plots[0] += 1
        output = next(iter(self._task.output_features.values()))
        for feature in features:
            f = self._task.input_features[feature]
            x = self._results[feature]["x"]
            if clustering:
                if cluster_param is None:
                    cluster_param = {}
                _, _, projection = self._cluster(f, **cluster_param)
                projection = projection.reshape(
                    (projection.shape[0], projection.shape[1], 1)
                )
            else:
                projection = self._results[feature]["y"]
            if n_subplot_cols > 0:
                plt.subplot(n_plots[0], n_plots[1], num_plot)
            elif num_plot > 1:
                plt.figure()
            num_plot += 1
            alpha = 1.0
            if projection.shape[0] > 10:
                alpha = 0.5
            if projection.shape[0] > 100:
                alpha *= 0.125
            if projection.shape[0] > 1000:
                alpha *= 0.125
            if f.data_type == DataType.ORDINAL:
                for i in range(projection.shape[0]):
                    plt.plot(x, projection[i, :, 0], "-r", alpha=alpha)
            else:
                for i in range(projection.shape[0]):
                    plt.plot(x, projection[i, :, 0], "-or", alpha=alpha)
                xtick_label = x
                if f.labels is not None:
                    xtick_label = [f.labels[t] for t in x]
                plt.xticks(x, xtick_label)
            plt.xlabel(f.name)
            # if f.data_type
            plt.ylabel(output.name)
            plt.title("Individual Conditional Expectation")

    def _cluster(
        self,
        feature: FeatureDescription,
        n_clusters=None,
        rel_tol=0.001,
        distance_threshold: float = 0.1,
        **sklearn_arg
    ):
        if feature.name in self._cluster_results.keys():
            logger.debug("Use cached cluster results for feature " + feature.name)
            res = self._cluster_results[feature.name]
            return res["cluster_no"], res["cluster_distance"], res["cluster_centers"]
        logger.debug(
            "Start functional clustering w.r.t. projection of feature " + feature.name
        )
        x = self._results[feature.name]["x"]
        projection = self._results[feature.name]["y"]
        cluster_no, cluster_distance, cluster_centers = (
            functional_clustering.agglomerative(
                projection[:, :, 0],
                n_clusters=n_clusters,
                rel_tol=rel_tol,
                distance_threshold=distance_threshold,
                **sklearn_arg
            )
        )
        self._cluster_results[feature.name] = {
            "cluster_no": cluster_no,
            "cluster_distance": cluster_distance,
            "cluster_centers": cluster_centers,
        }
        logger.debug(
            "Finished functional clustering w.r.t. projection of feature "
            + feature.name
        )
        return cluster_no, cluster_distance, cluster_centers


class ICE(BaseLocalExplainer):
    def __init__(self,
                 task: ValidationTask,
                 min_values=None,
                 max_values=None,
                 max_samples=None,
                 n_steps=20,
                 **kwargs):

        super().__init__(task,
                         min_values=min_values,
                         max_values=max_values,
                         max_samples=max_samples,
                         n_steps=n_steps,
                         **kwargs)

        if self.max_samples is None:
            self.max_samples = task.data.shape[0]
        else:
            self.max_samples = min(self.max_samples, task.data.shape[0])

        if task.output_dim() != 1:
            raise NotImplementedError("Outputdimension != 1 not implemented yet.")

    def _explain(self, x: np.ndarray) -> BaseExplanation:
        if x is None:
            x = self.task.data[
                np.random.choice(
                    self.task.data.shape[0], size=self.max_samples, replace=False
                ),
                :,
            ]
        # return super().explain(x)

    def _explain(self, x: np.ndarray) -> ICEExplanation:
        result = ICEExplanation(self.task, self._hashkey)
        for feature in self.task.input_features.values():
            self._project(x, feature, result)
        return result

    def _project(
        self,
        x: np.ndarray,
        feature: FeatureDescription,
        result: ICEExplanation,
    ):
        if feature.data_type != DataType.ORDINAL:
            return self._project_discrete(x, feature)
        WrongDataType.check_equality(feature, DataType.ORDINAL)
        if (self.min_values is None) or (feature.name not in self.min_values.keys()):
            min_value = self.task.data[:, feature.column].min()
        else:
            min_value = self.min_values[feature.name]
        if (self.max_values is None) or (feature.name not in self.max_values.keys()):
            max_value = self.task.data[:, feature.column].max()
        else:
            max_value = self.max_values[feature.name]
        data = x  # x[:, feature.column]
        projection = np.linspace(min_value, max_value, self.n_steps)
        projected_values = np.empty(
            (data.shape[0], projection.shape[0], self.task.output_dim())
        )
        for i in range(data.shape[0]):
            x = np.repeat(data[i : i + 1, :], repeats=projection.shape[0], axis=0)
            x[:, feature.column] = projection
            y = self.task.predict(x)
            if len(y.shape) == 1:
                y = y.reshape((-1, 1))
            projected_values[i, :, :] = y
        return result.add_result(feature, projection, projected_values)

    def _project_discrete(
        self, x: np.ndarray, feature: FeatureDescription, result: ICEExplanation
    ):
        data = x
        if feature.data_type == DataType.ORDINAL_DISCRETE:
            projection = np.unique(data[:, feature.column])
            projected_values = np.empty(
                (data.shape[0], projection.shape[0], self.task.output_dim())
            )
            for i in range(data.shape[0]):
                x = np.repeat(data[i : i + 1, :], repeats=projection.shape[0], axis=0)
                x[:, feature.column] = projection
                y = self.task.predict(x)
                if len(y.shape) == 1:
                    y = y.reshape((-1, 1))
                projected_values[i, :, :] = y
            return result.add_result(feature, projection, projected_values)

    def _get_total_figures(self):
        figures_total = 0
        for f in self.task.input_features.values():
            if f.data_type == DataType.ORDINAL.value:
                figures_total += 1

    def plot(self, features: List[str] = None):
        parameter_type = "sub_plot"
        plot_param = UtilsClass.set_config(parameter_type)
        sub_plt = SubPlots(
            len(self.task.get_input_cols_ordinal()) + 1,
            plot_param["max_subplot_columns"],
            plot_param["use_subplots"],
            plot_param["new_figure"],
        )
        output = next(iter(self.task.output_features.values()))
        if features is None:
            features = []
            for f in self.task.input_features.values():
                if f.data_type != DataType.ONE_HOT_ENCODED:
                    features.append(f.name)
        for feature in features:
            sub_plt.figure()
            f = self.task.input_features[feature]
            x, projection = self._project(self.task.data, f, self.task.y_pred)
            if f.data_type == DataType.ORDINAL:
                for i in range(projection.shape[0]):
                    plt.plot(x, projection[i, :, 0], "-r", alpha=0.02)
            else:
                for i in range(projection.shape[0]):
                    plt.plot(x, projection[i, :, 0], "-or", alpha=0.02)
                xtick_label = x
                if f.labels is not None:
                    xtick_label = [f.labels[t] for t in x]
                plt.xticks(x, xtick_label)

            if self.examples is not None:
                label = "examples"
                if f.data_type == DataType.ORDINAL:
                    for i in self.examples.points:
                        plt.plot(x, projection[i, :, 0], "-b", alpha=1.0, label=label)
                        label = None
                else:
                    for i in self.examples.points:
                        plt.plot(x, projection[i, :, 0], "-bo", alpha=1.0, label=label)
                        label = None
                plt.legend()
            plt.xlabel(f.name)
            # if f.data_type
            plt.ylabel(output.name)

    # def _cluster(self, feature: FeatureDescription):
    #     logger.debug('Start functional clustering w.r.t. projection of feature ' + feature.name)
    #     cache = self.cache.get(feature.name, {})
    #     if 'cluster_distance' not in cache.keys():
    #         logger.debug('Clustering results found in cache.')
    #         x, projection = self._project(feature)
    #         cluster_no, cluster_distance, cluster_centers = functional_clustering.agglomerative(projection[:,:,0],
    #                                                                                             **self.param.agg_cluster_param)
    #         cache['cluster_no'] = cluster_no
    #         cache['cluster_distance'] = cluster_distance
    #         cache['cluster_centers'] = cluster_centers
    #     logger.debug('Finished functional clustering w.r.t. projection of feature ' + feature.name)
    #     return cache['cluster_no'], cache['cluster_distance'], cache['cluster_centers']

    # def plot_cluster(self):
    #     output = next(iter(self.task.output_features.values()))
    #     num_plot = 1
    #     for f in self.task.input_features.values():
    #         if f.data_type == DataType.ORDINAL.value:
    #             #if f.name not in self.cluster.keys():
    #             x, projection = self._project(f)
    #             cluster_no, cluster_distance, cluster_centers = self._cluster(f)
    #             #plt.subplot(n_plots[0], n_plots[1], num_plot)
    #             num_plot += 1
    #             plt.figure()
    #             for i in range(cluster_centers.shape[0]):
    #                 plt.plot(x, cluster_centers[i], '-r')
    #             plt.xlabel(f.name)
    #             plt.ylabel(output.name)
    #             plt.title('Cluster centers for ICE on feature ' + f.name)
    #             #plt.subplot(n_plots[0], n_plots[1], num_plot)
    #             plt.figure()
    #             num_plot += 1
    #             depp = cluster_distance.max(axis=1)
    #             q=np.quantile(depp, 0.01)
    #             selection = projection[depp<q,:,0]
    #             for i in range(selection.shape[0]):
    #                 plt.plot(x, selection[i],'-r')
    #             plt.xlabel(f.name)
    #             plt.ylabel(output.name)
    #             plt.title('ICE plots, low similarity to cluster centers on feature ' + f.name)
    #             #plt.subplot(n_plots[0], n_plots[1], num_plot)
    #             plt.figure()
    #             num_plot += 1
    #             plt.title('Distribution of similarity of ICE plots on feature ' + f.name)
    #             for i in range(cluster_centers.shape[0]):
    #                 plt.hist(cluster_distance.max(axis=1), bins=20, density=True,)
    #             plt.xlabel('similarity to next cluster center')


if __name__ == "__main__":
    import sloth
    from sklearn.datasets import load_diabetes
    from sklearn import linear_model

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
    # validation_task = sloth.datasets.credit_default.get(dataset=2, model=0)
    ice = sloth.ICE(validation_task)
    ice.explain(validation_task.data[:50, :])  # TODO ice funktioniert nicht
