#######
Trainer
#######

EvaluateModel
=============

The EvaluateModel class takes in an untrained classifier, features, a target variable and a number of trials.  For each trial, a new split is made between testing and training data using a different random seed.  Then the relevant loss measures are calculated for each train test split.  The advantage of this method is seeing how sensitive a machine learning model with a given set of hyper parameters is to the training data supplied to it.  

For instance, if we are dealing with a classification problem and the precision, recall, and f1-score vary wildly, then we can be sure that the model is very sensitive to training data.  Additionally, we have a realistic sense of how well the model will perform on average, rather than in specific.  

All models are sensitive to the train/test split of the data to at least some degree, because they are (hopefully) faithfully reproducing the pattern found in the training data.  If some critical motif in the data is found in the test suite and not in the training, then this can be discovered by looking at which splits produce extremely bad measures.  This is of course a very nuanced use of the class.

In general, the best use of this class is to get a rough sense of how much a set of relevant loss functions will score a given machine learning model and set of hyper parameters.  For this reason, this is best used after hyperparameter tuning, perhaps testing with a holdout set thrown in the mix that wasn't used during hyper parameter tuning.  

Examples using EvaluateModel
============================

Here is an example using EvaluateModel on a toy classification problem::

	import warnings
	warnings.filterwarnings('ignore')

	from honest_ml.trainer import EvaluateModel
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

Here is an example of model_instances::

	{'0': {'precision': 0.944954128440367,
	  'recall': 0.865546218487395,
	  'f1-score': 0.9035087719298246,
	  'support': 119},
	 '1': {'precision': 0.8241758241758241,
	  'recall': 0.9259259259259259,
	  'f1-score': 0.872093023255814,
	  'support': 81},
	 'accuracy': 0.89,
	 'macro avg': {'precision': 0.8845649763080956,
	  'recall': 0.8957360722066605,
	  'f1-score': 0.8878008975928193,
	  'support': 200},
	 'weighted avg': {'precision': 0.8960389152132271,
	  'recall': 0.89,
	  'f1-score': 0.8907853937168503,
	  'support': 200},
	 'mask': array([ True,  True,  True, ...,  True,  True, False]),
	 'seed': 0,
	 'hyperparameters': {'C': 1.0,
	  'class_weight': None,
	  'dual': False,
	  'fit_intercept': True,
	  'intercept_scaling': 1,
	  'l1_ratio': None,
	  'max_iter': 100,
	  'multi_class': 'auto',
	  'n_jobs': None,
	  'penalty': 'l2',
	  'random_state': None,
	  'solver': 'lbfgs',
	  'tol': 0.0001,
	  'verbose': 0,
	  'warm_start': False},
	 'coef': array([[-0.09676706, -0.06259625,  0.03854064, -0.01988145, -0.03094951,
	          0.10734214,  0.05753582,  0.02321389, -0.05465297, -0.09141957,
	         -0.00392064, -0.32287119, -0.0666573 ,  0.07730165, -0.04026514,
	         -0.13186327,  0.02018894,  0.09521878,  0.01950879, -0.00566804,
	         -0.03334937,  0.03858804,  0.01047294, -0.0586396 ,  0.0295838 ,
	          0.01036426, -0.10592627,  0.06334281, -0.01646385, -0.08390779,
	         -0.00086186,  0.09208098,  0.06384728, -0.10523845,  0.04723001,
	         -0.02344517,  0.02810162,  0.01020396,  0.08977473, -0.02720635,
	         -0.04918006, -0.00447423,  0.00620328, -0.04005729,  0.06630182,
	         -0.06289338, -0.10858438,  0.08247678, -0.09550081,  0.02742673,
	          0.04816062, -0.04775787, -0.09390672, -0.04865164,  0.01400401,
	         -0.00238228, -0.02485218,  0.08137675,  0.04044883,  0.064263  ,
	          0.01840813, -0.00091731, -0.0287077 , -0.04312707, -0.08485293,
	          0.01058204, -0.08508349,  0.00258315, -0.12877512, -0.08236222,
	         -0.01701256,  0.00132286,  0.02480645,  0.13622381,  0.09167272,
	         -0.03295724,  0.01973968, -0.1196328 , -0.04922462, -0.05301064,
	          0.03037948,  0.02893391,  0.02708848,  0.06174348, -0.02857682,
	          0.15893328, -0.13167437, -0.11204371,  0.04979953,  0.02435652,
	          0.05545951, -0.10905302, -0.07886216,  0.00202794,  0.14029013,
	         -0.04551669, -0.06110873, -0.07522507, -0.08212976, -0.05911751]])}

in the above example, model_instances is a list.  Each of the elements is a dictionary.  The key '0' is the zeroth class from the target variable.  And the key '1' is the first class from the target variable.  You can see the precision, recall and f1-scores for each class, as well as all the other standard measure you'd get from classification_report from scikit-learn.

Additionally, you get the coefficients in coef.  As well as the hyperparameters used.  The loss functions are then visualized in honest_ml.visualizer.Visualizer.

RegressionTrainer
=================

In addition to EvaluateModel trainer comes with a regression specific training class that allows the user to fit the models in parallel as well as in sequence, as well as specifying a seed strategy.  

To choose how to train, simply call the fit method with keyword argument training="parallel".  

Example::

	from honest_ml.trainer import RegressionTrainer
	from sklearn.linear_model import LinearRegression
	from sklearn.datasets import make_regression
	import pandas as pd
	import numpy as np

	reg = LinearRegression()
	X, y = make_regression(
 	   n_samples=2000,  n_features=100,
	    n_informative=90,
	    random_state=0
	)
	X = pd.DataFrame(X)
	y = pd.Series(y)
	num_trials = 200
	test_size = 200

	regressor = RegressionTrainer(reg)
	model_instances = regressor.fit(
		X, y, num_trials, test_size, 
		seed_strategy="random",
		training="parallel"
	)

Additionally, notice the seed_strategy can be set to "random" or "sequential".  If it is set to random, then seeds are randomly selected from a random number generator.  Otherwise each seed is set sequentially from 0 up to the number of trials for each run.

Here the regressor also has a predict method::

	regressor.predict(X)

The predict method ensembles all the model instances and then takes the mean of all their predictions.  Additionally, you can ensemble just the top k models by with the following::
	
	regressor.predict(
		X, 
		k=0.1, 
		ensemble="top_k_percent"
	)

ClassificationTrainer
=====================

Just as there is a regression specific trainer, there is a classification specific trainer that takes the same parameters.  Below is a code example for how to use it::

	import warnings
	warnings.filterwarnings('ignore')

	from honest_ml.trainer import ClassificationTrainer
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

	classifier = ClassificationTrainer(clf)
	model_instances = classifier.fit(
		X, y, num_trials, test_size, 
		seed_strategy="random",
		training="parallel"
	)

As you can see the API is essentially equivalent.  The ClassificationTrainer also comes with a predict method that functions in the same way as the RegressionTrainer.  
