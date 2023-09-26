#######
Experiment
#######

Experiment
===========

The Experiment class acts like file handle around the EvaluateModel class, generating tracking information for each of your experiments.  With it you can log metadata about the evaluation, the model, and the number of trials.  Additionally, you can save the visualizations to the experiments directory that gets generated for ease of organization.

Examples using EvaluateModel
============================

Here is an example using Experiment with a toy classification problem::

	import warnings
	warnings.filterwarnings('ignore')

	from randomizer_ml.trainer import RegressionTrainer, ClassificationTrainer, EvaluateModel
	from randomizer_ml.visualizer import Visualizer
	from randomizer_ml.experiment import Experiment
	from sklearn.linear_model import LinearRegression
	from sklearn.datasets import make_regression
	from sklearn.linear_model import LogisticRegression
	from sklearn.datasets import make_classification
	import pandas as pd
	import numpy as np

	with Experiment("logistic_regression") as experiment:
	    clf = LogisticRegression()
	    X, y = make_classification(
	        n_samples=2000,  n_features=100,
	        n_informative=90, n_redundant=2,
	        random_state=0
	    )
	    X = pd.DataFrame(X)
	    y = pd.Series(y)
	    num_trials = 200

	    clf_eval = EvaluateModel("classification", clf, X, y, num_trials)
	    model_instances = clf_eval.fit_random("random")
	    experiment.log_model_instances(model_instances)
	    experiment.log_model(clf)
	    experiment.log_num_trials(num_trials)


	viz = Visualizer(
	    model_instances, 
	    "classification", 
	    coef_names=X.columns.tolist(),
	    output_dir="experiments/logistic_regression/"
	)

	viz.visualize_classification(
	    bins=len(model_instances),
	    show_plot=True,
	    save_plots=True,
	    formatting="png"
	)

	viz.visualize_coeficients(
	    bins=len(model_instances),
	    show_plot=True,
	    save_plots=True,
	    formatting="png"
	)