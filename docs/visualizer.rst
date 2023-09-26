##########
Visualizer
##########

The visualizer class is much lighter than the Trainer classes.  It just takes in model_instances and produces a visualization.

Of course, the visualizer class is just a suggestion.  You can certainly do your own visualizations, if you feel this class is lacking.

The first two examples below show a visualization of classification results.  That said, this works just as well for regression, using the visualize_regression method, which we will see in the third example.  

Example One::

	import warnings
	warnings.filterwarnings('ignore')

	from randomizer_ml.visualizer import Visualizer
	from randomizer_ml.trainer import EvaluateModel
	from sklearn.linear_model import LogisticRegression
	from sklearn.datasets import make_classification
	import pandas as pd
	import numpy as np

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

	viz = Visualizer(model_instances, "classification", coef_names=X.columns.tolist())
	viz.visualize_classification(bins=len(model_instances))

As you can see, the task is specified, the coeficient names are gathered and then visualization is rendered with visualize_classification.

Example Two::

	import warnings
	warnings.filterwarnings('ignore')

	from randomizer_ml.visualizer import Visualizer
	from randomizer_ml.trainer import EvaluateModel
	from sklearn.linear_model import LogisticRegression
	from sklearn.datasets import make_classification
	import pandas as pd
	import numpy as np

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

	viz = Visualizer(model_instances, "classification", coef_names=X.columns.tolist())
	viz.visualize_confusion_matrix()

Here we visualize the confusion_matrix.  This method only works for classification problems.  The reason it's useful to visualize the confusion matrix, as a scatter plot is to see whether or not the results are high for both precision and recall.  If the majority of cases are only high for one of the two loss measures, then the model isn't particularly good.  This can get lost when reviewing the results just from visualize_classification.

