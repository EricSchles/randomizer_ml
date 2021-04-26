from honest_ml.trainer import RegressionTrainer, ClassificationTrainer, EvaluateModel
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import pandas as pd

def test_regression_parallel_fit():
    lin_reg = LinearRegression()
    reg_trainer = RegressionTrainer(lin_reg)
    X, y = make_regression(
        n_samples=2000,  n_features=100, random_state=0
    )
    num_trials = 500
    test_size = 0.1
    model_instances = reg_trainer.fit(
        X, y, num_trials, test_size
    )
    assert len(model_instances) == num_trials

def test_regression_sequential_fit():
    lin_reg = LinearRegression()
    reg_trainer = RegressionTrainer(lin_reg)
    X, y = make_regression(
        n_samples=2000,  n_features=100, random_state=0
    )
    num_trials = 500
    test_size = 0.1
    model_instances = reg_trainer.fit(
        X, y, num_trials, test_size,
        training="sequential"
    )
    assert len(model_instances) == num_trials

def test_regression_sequential_sequential_seed_fit():
    lin_reg = LinearRegression()
    reg_trainer = RegressionTrainer(lin_reg)
    X, y = make_regression(
        n_samples=2000,  n_features=100, random_state=0
    )
    num_trials = 500
    test_size = 0.1
    model_instances = reg_trainer.fit(
        X, y, num_trials, test_size,
        training="sequential",
        seed_strategy="sequential"
    )
    assert len(model_instances) == num_trials

def test_regression_parallel_sequential_seed_fit():
    lin_reg = LinearRegression()
    reg_trainer = RegressionTrainer(lin_reg)
    X, y = make_regression(
        n_samples=2000,  n_features=100, random_state=0
    )
    num_trials = 500
    test_size = 0.1
    model_instances = reg_trainer.fit(
        X, y, num_trials, test_size,
        seed_strategy="sequential"
    )
    assert len(model_instances) == num_trials

def test_classification_parallel_fit():
    clf = LogisticRegression()
    clf_trainer = ClassificationTrainer(clf)
    X, y = make_classification(
        n_samples=2000,  n_features=100,
        n_informative=90, n_redundant=2,
        random_state=0
    )
    num_trials = 500
    test_size = 0.1
    model_instances = clf_trainer.fit(
        X, y, num_trials, test_size
    )
    assert len(model_instances) == num_trials

def test_classification_sequential_fit():
    clf = LogisticRegression()
    clf_trainer = ClassificationTrainer(clf)
    X, y = make_classification(
        n_samples=2000,  n_features=100,
        n_informative=90, n_redundant=2,
        random_state=0
    )
    num_trials = 500
    test_size = 0.1
    model_instances = clf_trainer.fit(
        X, y, num_trials, test_size,
        training="sequential"
    )
    assert len(model_instances) == num_trials

def test_classification_sequential_sequential_seed_fit():
    clf = LogisticRegression()
    clf_trainer = ClassificationTrainer(clf)
    X, y = make_classification(
        n_samples=2000,  n_features=100,
        n_informative=90, n_redundant=2,
        random_state=0
    )
    num_trials = 500
    test_size = 0.1
    model_instances = clf_trainer.fit(
        X, y, num_trials, test_size,
        training="sequential",
        seed_strategy="sequential"
    )
    assert len(model_instances) == num_trials

def test_classification_parallel_sequential_seed_fit():
    clf = LogisticRegression()
    clf_trainer = ClassificationTrainer(clf)
    X, y = make_classification(
        n_samples=2000,  n_features=100,
        n_informative=90, n_redundant=2,
        random_state=0
    )
    num_trials = 500
    test_size = 0.1
    model_instances = clf_trainer.fit(
        X, y, num_trials, test_size,
        seed_strategy="sequential"
    )
    assert len(model_instances) == num_trials

def test_evaluate_random():
    lin_reg = LinearRegression()
    X, y = make_regression(
        n_samples=2000,  n_features=100, random_state=0
    )
    X = pd.DataFrame(X)
    y = pd.Series(y)
    num_trials = 500
    test_size = 0.1
    reg_eval = EvaluateModel("regression", lin_reg, X, y, num_trials)
    model_instances = reg_eval.fit_random("random")
    assert len(model_instances) == num_trials
