from sklearn.datasets import load_diabetes
from sklearn import linear_model
import sloth, shap

#Gives examples of useage


x, y = load_diabetes(as_frame=True, return_X_y=True)
lin_regr = linear_model.LinearRegression().fit(x, y)

input_features = [sloth.OrdinalFeatureDescription(name=name) for name in x.columns if name != 'sex'] # all feature ordinal except sex
input_features.append(sloth.DiscreteOrdinalFeatureDescription(name='sex'))

output_feature = sloth.OrdinalFeatureDescription(name='disease progression', column=0)

validation_task = sloth.ValidationTask(input_features=input_features, output_features=output_feature,
                                       data=x, predict=lin_regr.predict, problemtype='regression')


############################################ global: ############################################


# ale explanation
ale = sloth.ALE(validation_task)
ale_exp = ale.explain()
ale_exp.plot()


# cohort shapley variance explanation
cs_var = sloth.CohortShapleyVariance(validation_task)
csv_exp = cs_var.explain()
# csv_exp.plot(features=['age', 'bmi', 's1', 's2', 's3', 's4', 's5'])
csv_exp.plot()


# m_plots explanation
mplot = sloth.MarginalPlots(validation_task)
mp_exp = mplot.explain()
mp_exp.plot()


# partial dependence plot explanation
pdp = sloth.PDP(validation_task)
pdp_exp = pdp.explain()
pdp_exp.plot()

# 2D partial dependence plot explanation
pdp2 = sloth.PDP2D(validation_task)
pdp2_exp = pdp2.explain()
pdp2_exp.plot(n_subplot_cols=6)



############################################ local: ############################################


# anchor explanation from different packages # Not working
# anchor = sloth.Anchors(validation_task)
# anchor_alibi = sloth.AnchorsAlibi(validation_task)
# anchor_exp = sloth.AnchorsExp(validation_task)

# cohort shapley explanation
coh_shap_explainer = sloth.CohortShapleyValues(validation_task)
c_sh = coh_shap_explainer.explain(validation_task.data[:100,:])
c_sh.plot_bar()


# ice explanation,
# ice_explainer = sloth.ICE(validation_task)
# ice = ice_explainer.explain(validation_task.data[:100,:]) # Not working
# ice.plot()


#  coordinate extreme points, extreme predictions, prototypes and criticism
examples = sloth.Examples(validation_task)
examples.compute_points()
print(examples.points)


# # lime explanation from lime package
# lime = sloth.Lime(validation_task, examples=examples)
# lime_expl = lime.explain(validation_task.data[:100,:])
# lime_expl.plot_by_features()
# lime.plot() #Not working


# shapley values from corr_shap package
shapley_explainer = sloth.ShapExplainer(validation_task)
shapley_explanation = shapley_explainer.explain(validation_task.data[:100,:])
shap.plots.beeswarm(shapley_explanation)

