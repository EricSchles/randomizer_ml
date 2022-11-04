from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn import ensemble
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
import random

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
            if "feature_importances_" in dir(self.model):
                report_dict['coef'] = self.model.feature_importances_
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

class EvaluateModelAndHyperParameters:
    def __init__(self,
                 model_type: str,
                 model: BaseEstimator,
                 data: pd.DataFrame,
                 target: pd.Series,
                 num_trials: int,
                 hyperparameters: list,
                 sufficient_compute: bool = False,
                 metrics: dict = {}):
        if model_type not in ["regression", "classification"]:
            raise Exception("model_type must be regression or classification")
        self.model_type = model_type
        self.model = model
        if self.is_pipeline():
            self.check_model_name()
        # rule of thumb for now
        if data.shape[0] * data.shape[1] > 50000:
            valid_tunable, compute_size = self.valid_tunable_hyperparameters(
                num_trials, hyperparameters
            )
            if (not valid_tunable) and (not sufficient_compute):
                raise Exception(f'''
                You are trying to make {compute_size} calculations
                with insufficient compute resources.
                If you feel you have reached this exception in error
                you can set sufficient_compute=True
                ''')
            if not valid_tunable:
                raise Warning(f'''
                You are trying to make {compute_size} calculations.
                Are you sure you have enough compute?
                ''')
        self.tunable_hyperparameters = hyperparameters
        self.hyperparameters = model.get_params()
        for tunable_param in self.tunable_hyperparameters:
            if isinstance(self.hyperparameters[tunable_param], str):
                raise Exception('String hyperparameters not supported yet')
        self.data = data
        self.target = target
        self.num_trials = num_trials
        self.metrics = metrics

    def fuzzer(self, value, size):
        '''
        Fuzzes a given parameter value to a 100 random values
        '''
        scale = abs(value)
        new_values = np.random.normal(0, scale * 3, size=size)
        # ensures the original value is in the array
        new_values = np.append(0, new_values)
        new_values += value
        return list(new_values)

    def valid_tunable_hyperparameters(self, num_trials, hyperparameters):
        compute_size = num_trials * len(hyperparameters) * 40
        return compute_size < 1000, compute_size
    
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

    def get_models_to_fit(self):
        base_model = self.model
        base_hp = self.hyperparameters
        param_dict = {}.fromkeys(self.tunable_hyperparameters)

        for param in self.tunable_hyperparameters:
            param_dict[param] = []
        for param in self.tunable_hyperparameters:
            value = base_hp[param]
            if isinstance(value, float):
                param_dict[param] += self.fuzzer(value, 10)
                param_dict[param] += self.fuzzer(value/10, 10)
                param_dict[param] += self.fuzzer(value/100, 10)
                param_dict[param] += self.fuzzer(value/1000, 10)
            if isinstance(value, int):
                param_dict[param] += self.fuzzer(value, 10)
                param_dict[param] += self.fuzzer(value * 3, 10)
                param_dict[param] += self.fuzzer(value * 5, 10)
                param_dict[param] += self.fuzzer(value * 10, 10)
            if isinstance(value, bool):
                param_dict[param] += [True, False]
        models = []
        for param in self.tunable_hyperparameters:
            for value in param_dict[param]:
                tmp_hp = base_hp
                tmp_model = base_model
                tmp_hp[param] = value
                try:
                    tmp_model.set_params(**tmp_hp)
                    models.append(tmp_model)
                except:
                    continue
        return models
                
    def _fit(self, results, seed, X_train, X_test, y_train, y_test):
        models = self.get_models_to_fit()
        for model in models:
            hyperparameters = model.get_params()
            try:
                model.fit(X_train, y_train)
            except:
                # invalid parameter configuration
                continue
            y_pred = model.predict(X_test)
            report_dict = self.report(y_test, y_pred)
            report_dict["mask"] = self._get_mask(y_train, self.data.shape[0])
            report_dict["seed"] = seed
            report_dict["hyperparameters"] = hyperparameters
            if self.is_pipeline():
                if 'coef_' in dir(model.named_steps['model']):
                    report_dict['coef'] = model.named_steps['model'].coef_
            else:
                if "coef_" in dir(model):
                    report_dict['coef'] = model.coef_
                if "feature_importances_" in dir(model):
                    report_dict['coef'] = model.feature_importances_
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

class GeneticAlgorithm:
    def __init__(
            self,
            models: list,
            stacking_model,
            problem_type: str,
            initial_hyperparameters: list,
            stacking_initial_hyperparameters: dict
    ):
        '''
        Parameters
        ----------
        * models : list - a list of uninitialized models.  
        Must follow the scikit-learn api.
        
        * stacking_model : scikit-learn model to use for stacking.
        
        * problem_type : str - regression or classification.
        Used to determine the stacking model to use.
                
        * initial hyperparameters : list of dictionaries.
        The initial hyperparameters to use for each model type.
        Passed to each model as Model(**initial_hyperparameters)
        '''
        self.models = models
        self.stacking_model = stacking_model
        self.initial_hyperparameters = initial_hyperparameters
        self.problem_type = problem_type
        self.stacking_initial_hyperparameters = stacking_initial_hyperparameters
        
    def initialize_population(self, size_per_model):
        '''
        Initialize a random population.

        Parameters
        ----------
        * size_per_model : list - the number of models to train,
        per model type.
        '''
        population = []
        for index, model in enumerate(self.models):
            for _ in range(size_per_model[index]):
                population.append(
                    model(**self.initial_hyperparameters[index])
                )
        return population
    
    def selection(self, X, y, test_size, seed, population, loss, top_k, higher_is_better=True):
        '''
        select k best members of the population for
        breeding based on loss.

        Parameters
        ----------
        * population : list - the population of models to
        select from.
        * loss : func - the loss function to use to select models.
        * top_k : int or float - if int, take the top k models (with 
        the best score).  If float, take the top percentage of models.
        if float, must be 0 < x <= 1.0
        * higher_is_better - bool.  Default is true.  For measures like
        mean_squared_error, or cross_entropy_loss this is false.  For measures
        like precision, recall or f1_score this is true.
        '''
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
        score_model = []
        for model in population:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = loss(y_test, y_pred)
            score_model.append([score, model])
        if isinstance(top_k, float):
            top_k = int(len(population) * top_k)
        score_model = sorted(
            score_model,
            key=lambda t: t[0],
            reverse=higher_is_better
        )[:top_k]
        return [_model[1] for _model in score_model]

    def fuzzer(self, value):
        '''
        Fuzzes a given parameter value to a 100 random values
        '''
        scale = abs(value)
        new_values = np.random.normal(0, scale * 3, size=100)
        # ensures the original value is in the array
        new_values = np.append(0, new_values)
        return new_values + value

    def loss_is_close(self, loss, y, loss_delta, higher_is_better):
        loss_min = loss(5, 5)
        if higher_is_better:
            assert np.allclose(loss_min, 1)
        else:
            assert np.allclose(loss_min, 0)
        loss_max = loss(1e-5, 1e10)
        if higher_is_better and np.allclose(loss_max, 0):
            return (loss_delta < 0.01)
        elif (not higher_is_better) and (loss_max > 10):
            return (loss_delta/y.mean() < 1)
        else:
            return loss_delta < 0.1
        
    def tune_hyperparameters(
            self, X, y, test_size, seed, loss, hyperparameters, higher_is_better
    ):
        # consider gradient update for parameters here.
        new_hyperparameters = {
            key: hyperparameters[key]
            for key in hyperparameters
        }

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
        original_model = self.stacking_model(**hyperparameters)
        original_model.fit(X_train, y_train)
        y_pred = original_model.predict(X_test)
        original_loss = loss(y_test, y_pred)
        for param in hyperparameters:
            value = hyperparameters[param]
            if isinstance(value, int) or isinstance(value, float):
                new_values = self.fuzzer(value)
                losses = []
                for index, new_value in enumerate(new_values):
                    new_hyperparameters[param] = new_value
                    model = self.stacking_model(
                        **new_hyperparameters
                    )
                    model.fit(X_trian, y_train)
                    y_pred = model.predict(X_test)
                    new_loss = loss(y_true, y_pred)
                    loss_delta = abs(original_loss - new_loss)
                    if not self.loss_is_close(loss, y, loss_delta, higher_is_better):
                        losses.append([new_loss, index])
                    # do binning on losses, if losses are 'close'
                    # to original parameter, don't include.
                losses = sorted(
                    losses,
                    key=lambda t: t[0],
                    reverse=higher_is_better
                )
                
                best_new_value = new_values[losses[0][1]]
                
                hyperparameters[param] = best_new_value
        return hyperparameters

    def crossover_and_mutate(
            self, X, y, test_size, seed, loss, population,
            breeding_rate, mutation_rate,
            stacking_model_hp,
            higher_is_better=True
    ):
        '''
        Combine two parents to create new model.
        This is where model stacking happens.
        Mutation occurs in the final model, 
        stacked ontop of the individual models.
        '''
        
        p = int((breeding_rate) * 10000)
        q = int((1 - breeding_rate) * 10000)
        should_breed = [True for _ in range(p)]
        shouldnt_breed = [False for _ in range(q)]
        should_breed += shouldnt_breed

        p = int((mutation_rate) * 10000)
        q = int((1 - mutation_rate) * 10000)
        should_mutate = [True for _ in range(p)]
        shouldnt_mutate = [False for _ in range(q)]
        should_mutate += shouldnt_mutate

        model_pairs = []
        original_stacking_model_hp = self.stacking_initial_hyperparameters
        new_stacking_model_hp = tune_hyperparameters(
            X, y, test_size, seed, loss, stacking_model_hp,
            higher_is_better=higher_is_better
        )

        for index_a, model_a in enumerate(population):
            for index_b, model_b in enumerate(population):
                if index_a == index_b:
                    continue
                if random.choice(should_breed):
                    model_pairs.append(
                        [model_a, model_b]
                    )
        if self.problem_type == "regression":
            for model_pair in model_pairs:
                if random.choice(should_mutate):
                    population.append(
                        ensemble.StackingRegressor(
                            estimators=model_pair,
                            final_estimator=self.stacking_model(
                                **new_stacking_model_hp
                            ),
                            passthrough=True
                        )
                    )
                else:
                    population.append(
                        ensemble.StackingRegressor(
                            estimators=model_pair,
                            final_estimator=self.stacking_model(
                                **original_stacking_model_hp
                            ),
                            passthrough=True
                        )
                    )
        if self.problem_type == "classification":
            for model_pair in model_pairs:
                if random.choice(should_mutate):
                    population.append(
                        ensemble.StackingClassifier(
                            estimators=model_pair,
                            final_estimator=self.stacking_model(
                                **new_stacking_model_hp
                            ),
                            passthrough=True
                        )
                    )
                else:
                    population.append(
                        ensemble.StackingClassifier(
                            estimators=model_pair,
                            final_estimator=self.stacking_model(
                                **original_stacking_model_hp
                            ),
                            passthrough=True
                        )
                    )                    
        return (
            population,
            new_stacking_model_hp
        )
        
    def is_stacked(self, model):
        model_type = str(type(model))
        return "stacking" in model_type

    def crossover(self, population, breeding_rate):
        '''
        Combine two parents to create new model.
        This is where model stacking happens.
        '''
        p = int((breeding_rate) * 10000)
        q = int((1 - breeding_rate) * 10000)
        should_breed = [True for _ in range(p)]
        shouldnt_breed = [False for _ in range(q)]
        should_breed += shouldnt_breed
        model_pairs = []

        for index_a, model_a in enumerate(population):
            for index_b, model_b in enumerate(population):
                if index_a == index_b:
                    continue
                if random.choice(should_breed):
                    model_pairs.append(
                        [model_a, model_b]
                    )
        if self.problem_type == "regression":
            for model_pair in model_pairs:
                population.append(
                    ensemble.StackingRegressor(
                        estimators=model_pair,
                        final_estimator=self.stacking_model(
                            **self.stacking_initial_hyperparameters
                        ),
                        passthrough=True
                    )
                )
        if self.problem_type == "classification":
            for model_pair in model_pairs:
                population.append(
                    ensemble.StackingClassifier(
                        estimators=model_pair,
                        final_estimator=self.stacking_model(
                            **self.stacking_initial_hyperparameters
                        ),
                        passthrough=True
                    )
                )
            
        return population
    
    def mutate(
            self,
            population,
            mutation_rate,
            X, y,
            test_size,
            seed, loss,
            higher_is_better=True):
        '''
        Mutate the hyperparameters of individual parents 
        before combining.
        '''
        p = int((mutation_rate) * 10000)
        q = int((1 - mutation_rate) * 10000)
        should_mutate = [True for _ in range(p)]
        shouldnt_mutate = [False for _ in range(q)]
        should_mutate += shouldnt_mutate

        new_population = []
        for model in population:
            if self.is_stacked(model):
                hyperparameters = model.get_params()
                tunable_hyperparameters = {}
                for param in hyperparameters:
                    if "final" in param:
                        tunable_hyperparameters[param] = hyperparameters[param]
                tunable_hyperparameters = self.tune_hyperparameters(
                    X, y,
                    test_size,
                    seed, loss,
                    tunable_hyperparameters,
                    higher_is_better
                )
                for param in tunable_hyperparameters:
                    hyperparameters[param] = tunable_hyperparameters[param]
            else:
                hyperparameters = model.get_params()
                hyperparameters = self.tune_hyperparameters(
                    X, y,
                    test_size,
                    seed, loss,
                    hyperparameters,
                    higher_is_better
                )
            model.set_params(**hyperparameters)
            new_population.append(model)
        return new_population
        

    def get_model_population(
            self, X, y, test_size, seed, loss, size_per_model,
            breeding_rate=0.1, mutation_rate=0.05,
            number_of_generations=50,
            strategy="crossover", higher_is_better=True
    ):
        '''
        Strategies:
        * crossover - only do cross over
        * mutate_crossover - mutate parents, then cross over
        * mutate_crossover_mutate - mutate parents, mutate stacked learner, then cross over.

        Steps:
        '''
        population = self.initialize_population(size_per_model)
        stacking_model_hp = self.stacking_initial_hyperparameters
        for _ in range(number_of_generations):
            if strategy == "crossover":
                population = self.crossover(population, breeding_rate)
            if strategy == "mutate_crossover":
                population = self.mutate(
                    population,
                    mutation_rate,
                    X, y,
                    test_size,
                    seed, loss,
                    higher_is_better=higher_is_better
                )
                population = self.crossover(population, breeding_rate)
            if strategy == "mutate_crossover_mutate":
                population = self.mutate(
                    population,
                    mutation_rate,
                    X, y,
                    test_size,
                    seed, loss,
                    higher_is_better=higher_is_better
                )
                population, stacking_model_hp = self.crossover_and_mutate(
                    self, X, y, test_size, seed,
                    loss, population,
                    breeding_rate, mutation_rate,
                    stacking_model_hp,
                    higher_is_better=True
                )
                
            population = self.selection(
                X, y, test_size, seed,
                population, loss,
                top_k, higher_is_better=True
            )
        return population
        
class GeneticBoosting(BaseTrainer, GeneticAlgorithm):
    def __init__(
            self,
            models: list,
            stacking_model,
            problem_type: str,
            initial_hyperparameters: list,
            stacking_initial_hyperparameters: dict
    ):
        self.models = models
        self.stacking_model = stacking_model
        self.initial_hyperparameters = initial_hyperparameters
        self.problem_type = problem_type
        self.stacking_initial_hyperparameters = stacking_initial_hyperparameters

    def _fit_parallel(
            self,
            X, y, test_size, seed, loss, size_per_model,
            breeding_rate, mutation_rate,
            number_of_generations,
            strategy, higher_is_better
    ):
        num_cpus = cpu_count()
        with concurrent.futures.ProcessPoolExecutor(num_cpus) as pool:
            seeds = []
            futures = []
            model_instances = []
            for seed in range(num_trials):
                if seed_strategy == "random":
                    seed, seeds = get_random_seed(seeds)
                future = pool.submit(
                    self.genetic_fit,
                    X, y, test_size, seed, loss,
                    size_per_model, breeding_rate,
                    mutation_rate,
                    number_of_generations,
                    strategy, higher_is_better
                )
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                model_instances.append(future.result())
        return model_instances

    def genetic_fit(
            self,
            X, y, test_size, seed, loss, size_per_model,
            breeding_rate, mutation_rate,
            number_of_generations,
            strategy, higher_is_better
    ):
        return self.get_model_population(
            X, y, test_size, seed, loss,
            size_per_model,
            breeding_rate=breeding_rate,
            mutation_rate=mutation_rate,
            number_of_generations=number_of_generations,
            strategy=strategy,
            higher_is_better=higher_is_better
        )


    def fit(
            self, X, y,
            num_trials,
            test_size,loss,
            size_per_model,
            breeding_rate=0.1,
            mutation_rate=0.05,
            number_of_generations=50,
            strategy="crossover",
            higher_is_better=True,
            seed_strategy="random"
    ):
        population = self._fit_parallel(
            X, y, test_size, seed, loss, size_per_model,
            breeding_rate, mutation_rate,
            number_of_generations,
            strategy, higher_is_better, seed_strategy
        )

        # boosting goes here - pass population to boosting
        model_instances = self._fit_sequential(
            X, y, test_size, num_trials, seed_strategy
        )
        
        self.model_instances = model_instances
        return model_instances


# gradient step:
# look ahead to fit the best marginal model
# look 'N' ahead to fit the best marginal models.

# deep stacking by generating synthetic data to train on, this ensures no data leakage.
# make sure synthetic data isn't literally in the test set to ensure quality.
