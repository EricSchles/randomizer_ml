from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn._loss.loss import (
    HalfGammaLoss,
    HalfPoissonLoss,
    HalfSquaredError,
    HalfTweedieLoss,
    HalfTweedieLossIdentity,
    HalfBinomialLoss,
    HalfMultinomialLoss
)

from sklearn.base import clone
import numpy as np
from multiprocessing import cpu_count
import concurrent.futures
import pandas as pd
import numpy as np
import operator

def get_random_seed(seeds):
    if seeds == []:
        seed = 0
    else:
        seed = seeds[0]
    while seed in seeds:
        seed = int(np.random.uniform(0, 1000000))
    seeds.append(seed)
    return seed, seeds

def get_sequential_mask(num_rows: int, pivot: int):
    first = np.full(pivot, True)
    remainder = num_rows - pivot
    second = np.full(remainder, False)
    return np.concatenate([first, second])

def mean(array):
    return sum(array)/len(array)

def get_counts(array):
    elements = list(set(array))
    counts = {}.fromkeys(elements, 0)
    for element in elements:
        counts[element] += 1
    return counts

def mode(array):
    counts = get_counts(array)
    return max(counts.items(), key=operator.itemgetter(1))[0]

def get_minority_class(array):
    counts = get_counts(array)
    return min(counts.item(), key=operator.itemgetter(1))[0]

class EvaluateModel:
    def __init__(self,
                 model_type: str,
                 model: BaseEstimator,
                 data: pd.DataFrame,
                 target: pd.Series,
                 num_trials: int,
                 metrics: dict = {}):
        if model_type not in ["regression", "classification"]:
            raise Exception("model_type must be regression or classification")
        self.model_type = model_type
        self.model = model
        if self.is_pipeline():
            self.check_model_name()
        self.hyperparameters = model.get_params()
        self.data = data
        self.target = target
        self.num_trials = num_trials
        self.metrics = metrics

    def _get_mask(self, y_train: pd.Series, num_rows: int):
        mask = np.full(num_rows, False)
        mask[y_train.index] = True
        return mask

    def is_pipeline(self):
        dummy_pipeline = Pipeline(steps=[
            ('dummy regressor', LogisticRegression())
        ])
        return type(self.model) == type(dummy_pipeline)

    def check_model_name(self):
        if 'model' != list(self.model.named_steps.keys())[-1]:
            raise Exception("model must be named 'model' in the pipeline")

    def custom_report(self, y_test, y_pred):
        return {
            metric_name: metric(y_test, y_pred)
            for (metric_name, metric) in self.metrics.items()
        }
            
    def report(self, y_test, y_pred):
        if self.metrics:
            report_dict = self.custom_report(y_test, y_pred)
        if self.model_type == "classification":
            report_dict = classification_report(y_test, y_pred, output_dict=True)
        elif self.model_type == "regression":
            report_dict = self.regression_report(y_test, y_pred)
        return report_dict
            
    def regression_report(self, y_test, y_pred):
        return {
            "mse": metrics.mean_squared_error(y_test, y_pred),
            "max_error": metrics.max_error(y_test, y_pred),
            "mae": metrics.mean_absolute_error(y_test, y_pred)
        }
        
    def _fit(self, results, seed, X_train, X_test, y_train, y_test):
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        report_dict = self.report(y_test, y_pred)
        report_dict["mask"] = self._get_mask(y_train, self.data.shape[0])
        report_dict["seed"] = seed
        report_dict["hyperparameters"] = self.hyperparameters
        if self.is_pipeline():
            if 'coef_' in dir(self.model.named_steps['model']):
                report_dict['coef'] = self.model.named_steps['model'].coef_
        else:
            if "coef_" in dir(self.model):
                report_dict['coef'] = self.model.coef_
        results.append(report_dict)
        return results

    def is_valid_split(self, y_train, y_test):
        if self.model_type == "classification":
            target_classes = self.target.unique()
            target_classes = sorted(target_classes)
            train_classes = y_train.unique()
            train_classes = sorted(train_classes)
            test_classes = y_test.unique()
            test_classes = sorted(test_classes)
            return (
                (target_classes == train_classes) and
                (target_classes == test_classes)
            )
        else:
            return True
        
    def fit_random(self, seed_strategy, test_size=0.1, seeds_tried=[]):
        results = []
        if seed_strategy == "sequential":
            for seed in range(self.num_trials):
                X_train, X_test, y_train, y_test = train_test_split(
                    self.data, self.target,
                    test_size=test_size,
                    random_state=seed
                )
                if not self.is_valid_split(y_train, y_test):
                    continue
                results = self._fit(
                    results, seed, X_train, X_test, y_train, y_test
                )
        elif seed_strategy == "random":
            seeds = [] + seeds_tried
            for _ in range(self.num_trials):
                seed, seeds = get_random_seed(seeds)
                X_train, X_test, y_train, y_test = train_test_split(
                    self.data, self.target,
                    test_size=test_size,
                    random_state=seed
                )
                if not self.is_valid_split(y_train, y_test):
                    continue
                results = self._fit(
                    results, seed, X_train, X_test, y_train, y_test
                )
        else:
            raise Exception("Unknown seed strategy.")
        return results

    def _mask_split(self, train_mask):
        test_mask = ~train_mask
        X_train = self.data[train_mask]
        X_test = self.data[test_mask]
        y_train = self.target[train_mask]
        y_test = self.target[test_mask]
        return (
            X_train, X_test,
            y_train, y_test
        )

    def fit_sequential(self):
        results = []
        num_rows = self.data.shape[0]
        if num_rows < 20:
            raise Exception("Cannot do sequential fit with less than 20 data points")
        first_ten_percent = int(num_rows * 0.1)
        last_ten_percent = int(num_rows * 0.9)
        seed = None
        for pivot in range(first_ten_percent, last_ten_percent):
            train_mask = get_sequential_mask(data.shape[0], pivot)
            X_train, X_test, y_train, y_test = self._mask_split(train_mask)
            if not self.is_valid_split(y_train, y_test):
                continue
            results = self._fit(
                results, seed, X_train, X_test, y_train, y_test
            )
        return results            

class BaseTrainer:
    def __init__(self, model):
        self.model = model
        self.hyperparameters = self.model.get_params()
        self.fit_models = []
        self.model_instances = None

    def _fit_parallel(self, X, y, test_size, num_trials, seed_strategy):
        num_cpus = cpu_count()
        with concurrent.futures.ProcessPoolExecutor(num_cpus) as pool:
            seeds = []
            futures = []
            model_instances = []
            for seed in range(num_trials):
                if seed_strategy == "random":
                    seed, seeds = get_random_seed(seeds)
                future = pool.submit(
                    self._fit, seed,
                    test_size, X, y
                )
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                model_instances.append(future.result())
        return model_instances

    def _fit_sequential(self, X, y, test_size, num_trials, seed_strategy):
        seeds = []
        model_instances = []
        for seed in range(num_trials):
            if seed_strategy == "random":
                seed, seeds = get_random_seed(seeds)
            model_instances.append(
                self._fit(
                    seed, test_size, X, y
                )
            )
        return model_instances
        
    def fit(self, X, y, num_trials, test_size, seed_strategy="random", training="parallel"):
        if training == "parallel":
            model_instances = self._fit_parallel(
                X, y, test_size, num_trials, seed_strategy
            )
        if training == "sequential":
            model_instances = self._fit_sequential(
                X, y, test_size, num_trials, seed_strategy
            )
        self.model_instances = model_instances
        return model_instances


class RegressionTrainer(BaseTrainer):
    def _fit(self, seed, test_size, X, y, metric=None):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=seed
        )
        model_instance = clone(self.model)
        model_instance.fit(X_train, y_train)
        y_pred = model_instance.predict(X_test)
        if metric is None:
            return [
                model_instance,
                metrics.mean_squared_error(
                    y_test, y_pred
                )
            ]
        else:
            return [
                model_instance,
                metric(y_test, y_pred)
            ]


    def predict(self, X, k=0.1, ensemble="all"):
        """
        ensemble options:
        * all
        * top_k_percent
        """
        predictions = []
        if ensemble == "all":
            for model_instance in self.model_instances:
                model = model_instance[0]
                predictions.append(
                    model.predict(X)
                )
            return mean(predictions)
        elif ensemble == "top_k_pecent":
            model_instances = sorted(
                self.model_instances, key=lambda t: t[1]
            )
            end = int(len(model_instances) * k)
            for i in range(end):
                model = model_instances[i][0]
                predictions.append(
                    model.predict(X)
                )
            return mean(predictions)
            

class ClassificationTrainer(BaseTrainer):
    def _fit(self, seed, test_size, X, y, metric=None):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=seed
        )
        model_instance = clone(self.model)
        model_instance.fit(X_train, y_train)
        y_pred = model_instance.predict(X_test)
        self.y = y
        if metric is None:
            return [
                model_instance,
                classification_report(
                    y_pred, y_test, output_dict=True
                )
            ]
        else:
            return [
                model_instance,
                metric(y_test, y_pred)
            ]


    def _transpose_predictions(self, predictions):
        final_predictions = []
        for elem in np.array(predictions).T:
            final_predictions.append(
                mode(elem)
            )
        return np.array(final_predictions)
        
    def predict(self, X, k=0.9, accuracy="f1-score", ensemble="all"):
        """
        ensemble options:
        * all
        * top_k_percent
        * k_best_minority_class
        """
        predictions = []
        if ensemble == "all":
            for model_instance in self.model_instances:
                model = model_instance[0]
                predictions.append(
                    model.predict(X)
                )
            return self._transpose_predictions(predictions)
        elif ensemble == "top_k_pecent":
            model_instances = sorted(
                self.model_instances,
                key=lambda t: t[1]["macro avg"][accuracy]
            )
            end = int(len(model_instances) * k)
            for i in range(end, len(model_instances)):
                model = model_instances[i][0]
                predictions.append(
                    model.predict(X)
                )
            return self._transpose_predictions(predictions)
        elif ensemble == "k_best_minority_class":
            minority_class = str(get_minority_class(self.y))
            model_instances = sorted(
                self.model_instances,
                key=lambda t: t[1][minority_class][accuracy]
            )
            end = int(len(model_instances) * k)
            for i in range(end, len(model_instances)):
                model = model_instances[i][0]
                predictions.append(
                    model.predict(X)
                )
            return self._transpose_predictions(predictions)

class BaseGradientBoosting:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.hyperparameters = self.model.get_params()
        self.fit_models = []
        self.model_instances = None

    def _fit_sequential(
            self,
            X, y,
            test_size,
            num_trials,
            seed_strategy,
            out_of_bag_metric,
            learning_rate,
            base_loss,
            fit_intercept,
            sample_weight,
            l2_reg_strength
    ):
        seeds = []
        for seed in range(num_trials):
            if seed_strategy == "random":
                seed, seeds = get_random_seed(seeds)
            seeds.append(seed)
        
        return self._fit(
            self.model_name,
            seeds, test_size,
            X, y,
            out_of_bag_metric,
            learning_rate,
            base_loss,
            fit_intercept,
            sample_weight,
            l2_reg_strength
        )
        
    def fit(self,
            X, y,
            num_trials,
            test_size,
            out_of_bag_metric=metrics.mean_squared_error,
            learning_rate=1e-3,
            base_loss=None,
            fit_intercept=True,
            sample_weight=None,
            l2_reg_strength=0,
            seed_strategy="random"):
        model_instances = self._fit_sequential(
            X, y, test_size, num_trials, seed_strategy,
            out_of_bag_metric, learning_rate,
            base_loss, fit_intercept,
            sample_weight, l2_reg_strength
        )
        self.model_instances = model_instances
        return model_instances

    def predict(self, X):
        return sum([
            model_instance["model_instance"].predict(X)
            for model_instance in self.model_instances
        ])

class GradientBoostingRegressor(BaseGradientBoosting):
    loss_lookup = {
        "LinearRegression": LinearModelLoss
    }
    
    def _fit(
            self,
            model_name,
            seeds, test_size,
            X, y,
            out_of_bag_metric,
            learning_rate,
            base_loss,
            fit_intercept,
            sample_weight,
            l2_reg_strength
    ):
        self.model_instances = []
        self.out_of_bag_metrics = []
        self.fit_models = []
        residuals = y
        model_instances = []
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                X, residuals,
                test_size=test_size,
                random_state=seed
            )
            model_instance = clone(self.model)
            model_instance.fit(X_train, y_train)
            self.fit_models.append(
                model_instance
            )

            if base_loss is not None:
                # This method doesn't work yet, it also explicitly calculates
                # the gradient subject to the loss of the model, instead of just
                # taking the derivative of mean squared error implicitly, which
                # seems to be standard practice.
                LossModel = self.loss_lookup[model_name]
                loss_model = LossModel(
                    base_loss=base_loss,
                    fit_intercept=fit_intercept
                )
                #import code
                #code.interact(local=locals())
                loss, gradient = loss_model.loss_gradient(
                    model_instance.coef_, X_train, y_train,
                    sample_weight=sample_weight,
                    l2_reg_strength=l2_reg_strength
                )
                y_pred = model_instance.predict(X_test)
                out_of_bag_error = out_of_bag_metric(y_test, y_pred)
                self.out_of_bag_metrics.append(
                    out_of_bag_error
                )
                y_pred = model_instance.predict(X_train)
                residuals = y_pred - (learning_rate * gradient * loss)
            else:
                y_pred = model_instance.predict(X_test)
                out_of_bag_error = out_of_bag_metric(y_test, y_pred)
                self.out_of_bag_metrics.append(
                    out_of_bag_error
                )
                residuals = y - learning_rate * model_instance.predict(X)
                
            model_instances.append({
                "model_name": model_name,
                "seed": seed,
                "gradient": gradient,
                "loss": loss,
                "out_of_bag_error": out_of_bag_error,
                "coef": model_instance.coef_,
                "model_instance": model_instance,
                "learning_rate": learning_rate,
                "fit_intercept": fit_intercept,
                "l2_reg_strength": l2_reg_strength,
                "out_of_bag_metric": out_of_bag_metric,
                "sample_weight": sample_weight
            })
        return model_instances




