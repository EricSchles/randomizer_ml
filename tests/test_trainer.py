from honest_ml.trainer import RegressionTrainer, ClassificationTrainer
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

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
