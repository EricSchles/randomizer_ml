import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KernelDensity

# for classification
        
# we could learn a model of hyperparameters importance to a given measure
# there are several values for each hyperparameter set.  So we need to find
# the right problem.  But this is a possability.


# next up:
# add more plots, like https://towardsdatascience.com/stop-using-0-5-as-the-threshold-for-your-binary-classifier-8d7168290d44
# text above the plot is "Console output (1/1):"
# also consider adding a plot where hyperparameter is on the x-axis and the relevant sensitivity measure is on the y?
# also consider adding a beeswarm plot like shap: https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html
# this would work a little different, each of the hyperparameter sensitivities would be shown, without the context of how much the hyperparameter
# in question changed.  So you could see the full variance of sensitivity, but not the inflection points.
# then use the 2nd plot or the histograms to figure out where the cutoffs are.
# also consider the first chart as a way of capturing all of this, for like 2 or 3 of the hyperparameters at a time maybe?
# I don't think you could show all of them.

# also, consider making text analyses for all of these.  Just because everyone else likes visuals doesn't mean I have to.
class AnalyzeClassificationMeasures:
    def __init__(self):
        pass

    def hist_plot_score_groups(self):
        measures = list(self.score_groups["base"].keys())
        for measure in measures:
            for group in self.score_groups:
                dist = self.score_groups[group][measure]
                plt.hist(dist)
                plt.title(f"{group} - {measure}")
                plt.show()

    def fit(self, model_instances, X, y, model=LinearRegression(), correlation_function=stats.spearmanr):
        self.model_instances = model_instances
        self.hp_groups = self.group_model_instances(model_instances)
        self.hp_groups = self.sort_by_seed(self.hp_groups)
        self.classes = self.get_classes(y)
        self.hp_max_scores = self.describe_scores(
            np.max, self.hp_groups, self.classes
        )
        self.hp_min_scores = self.describe_scores(
            np.min, self.hp_groups, self.classes
        )
        self.hp_mean_scores = self.describe_scores(
            np.mean, self.hp_groups, self.classes
        )
        self.hp_median_scores = self.describe_scores(
            np.median, self.hp_groups, self.classes
        )
        self.hp_iqr_scores = self.describe_scores(
            stats.iqr, self.hp_groups, self.classes
        )
        self.hp_range_scores = self.get_range(
            self.hp_max_scores, self.hp_min_scores
        )
        self.group_diffs = self.difference_hyperparameters(
            self.hp_groups
        )
        self.score_groups = self.get_scores(
            self.hp_groups, self.classes
        )
        self.best_hyperparameters = self.get_best_hyperparameters(
            model_instances, y
        )
        self.get_groups(model=model, correlation_function=correlation_function)

    def get_groups(self, model=LinearRegression(), correlation_function=stats.spearmanr):
        self.model_groups = self.get_model_relationship(
            model=model
        )
        self.corr_groups = self.get_correlations(
            corr_func=correlation_function
        )
        self.var_groups = self.get_variances()
        self.concentration_groups = self.get_concentration()
        self.iqr_groups = self.get_iqrs()
        self.range_groups = self.get_range_groups()
        self.density_groups = self.get_density()        

    def coarse_grained_analysis(self):
        # fill this in with coarse analyses as defined below
        self.coarse_var = self.coarse_analyze_variance_sensitivity()
        self.coarse_concentration = self.coarse_analyze_concentration_sensitivity()
        self.coarse_iqr = self.coarse_analyze_iqr_sensitivity()
        self.coarse_range = self.coarse_analyze_range_sensitivity()
        self.coarse_density_ks = self.coarse_analyze_density_sensitivity_ks_2samp()
        self.coarse_density_cramer_von_mises = self.coarse_analyze_density_sensitivity_cramervonmises_2samp()
        self.coarse_corr = self.coarse_analyze_correlation_sensitivity()
        self.coarse_model = self.coarse_analyze_model_based_sensitivity()
        return {
            "coarse_variance_sensitivity": self.coarse_var,
            "coarse_concentration_sensitivity": self.coarse_concentration,
            "coarse_iqr_sensitivity": self.coarse_iqr,
            "coarse_range_sensitivity": self.coarse_range,
            "coarse_density_ks_2sample_sensitivity": self.coarse_density_ks,
            "coarse_density_cramer_von_mises_2sample_sensitivity": self.coarse_density_cramer_von_mises,
            "coarse_correlation_sensitivity": self.coarse_corr,
            "coarse_model_sensitivity": self.coarse_model
        }

    def fine_grained_analysis(self):
        # fill this in with fine analyses as defined below
        self.fine_var = self.fine_analyze_variance_sensitivity()
        self.fine_concentration = self.fine_analyze_concentration_sensitivity()
        self.fine_iqr = self.fine_analyze_iqr_sensitivity()
        self.fine_range = self.fine_analyze_range_sensitivity()
        self.fine_density_ks = self.fine_analyze_density_sensitivity_ks_2samp()
        self.fine_density_cramer_von_mises = self.fine_analyze_density_sensitivity_cramervonmises_2samp()
        self.fine_corr = self.fine_analyze_correlation_sensitivity()
        self.fine_model = self.fine_analyze_model_based_sensitivity()
        return {
            "fine_variance_sensitivity": self.fine_var,
            "fine_concentration_sensitivity": self.fine_concentration,
            "fine_iqr_sensitivity": self.fine_iqr,
            "fine_range_sensitivity": self.fine_range,
            "fine_density_ks_2sample_sensitivity": self.fine_density_ks,
            "fine_density_cramer_von_mises_2sample_sensitivity": self.fine_density_cramer_von_mises,
            "fine_correlation_sensitivity": self.fine_corr,
            "fine_model_sensitivity": self.fine_model
        }
        

    def group_model_instances(self, model_instances):
        hp_group = {}
        for model_instance in model_instances:
            hp_set = model_instance["hyperparameter_set"]
            if hp_set not in hp_group:
                hp_group[hp_set] = [model_instance]
            else:
                hp_group[hp_set].append(model_instance)
        return hp_group

    def describe_scores(self, measure, hp_groups, classes):
        hp_score = {}
        for group in hp_groups:
            hp_score[group] = {
                    "f1-score, macro-avg": measure([
                    model_instance["macro avg"]["f1-score"]
                    for model_instance in hp_groups[group]
                ]),
                    "precision, macro-avg": measure([
                    model_instance["macro avg"]["precision"]
                    for model_instance in hp_groups[group]
                ]),
                    "recall, macro-avg": measure([
                    model_instance["macro avg"]["recall"]
                    for model_instance in hp_groups[group]
                ])
            }
            for _class in classes:
                hp_score[group][f"f1-score, {_class}"] = measure([
                    model_instance[f"{_class}"]["f1-score"]
                    for model_instance in hp_groups[group]
                ])
                hp_score[group][f"precision, {_class}"] = measure([
                    model_instance[f"{_class}"]["precision"]
                    for model_instance in hp_groups[group]
                ])
                hp_score[group][f"recall, {_class}"] = measure([
                    model_instance[f"{_class}"]["recall"]
                    for model_instance in hp_groups[group]
                ])
        return hp_score

    def get_classes(self, y):
        return [str(elem) for elem in list(np.unique(y.ravel()))]

    def get_range(self, hp_max_scores, hp_min_scores):
        hp_range_scores = {}
        for group in hp_max_scores:
            hp_range_scores[group] = {}
            for measure in hp_max_scores[group]:
                hp_range_scores[group][measure] = (
                    hp_max_scores[group][measure] - hp_min_scores[group][measure]
                )
        return hp_range_scores

    def formula(self, max_scores, min_scores, mean_scores, median_scores, iqr_scores, range_scores):
        max_score = 0
        min_score = 0
        mean_score = 0
        median_score = 0
        iqr_score = 0
        range_score = 0
        for measure in max_scores:
            max_score += max_scores[measure]
            min_score += min_scores[measure]
            median_score += median_scores[measure]
            iqr_score += iqr_scores[measure]
            range_score += range_scores[measure]
        return (
            max_score + min_score + mean_score + median_score + 1/iqr_score + 1/range_score
        )

    def get_best_hyperparameters(self, model_instances, y):
        classes = self.classes
        hp_groups = self.hp_groups
        aggregated_groups = []
        for group in hp_groups:
            max_scores = self.hp_max_scores[group]
            min_scores = self.hp_min_scores[group]
            mean_scores = self.hp_mean_scores[group]
            median_scores = self.hp_median_scores[group]
            iqr_scores = self.hp_iqr_scores[group]
            range_scores = self.hp_range_scores[group]
            result = self.formula(
                max_scores, min_scores,
                mean_scores, median_scores,
                iqr_scores, range_scores
            )
            aggregated_groups.append(
                (group, result)
            )
        aggregated_groups = sorted(aggregated_groups, key=lambda t: t[1])
        return aggregated_groups[-1]

    def get_scores(self, hp_groups, classes):
        hp_score = {}
        for group in hp_groups:
            hp_score[group] = {
                "f1-score, macro-avg": [
                    model_instance["macro avg"]["f1-score"]
                    for model_instance in hp_groups[group]
                ],
                "precision, macro-avg": [
                    model_instance["macro avg"]["precision"]
                    for model_instance in hp_groups[group]
                ],
                "recall, macro-avg": [
                    model_instance["macro avg"]["recall"]
                    for model_instance in hp_groups[group]
                ],
                "seed": [
                    model_instance["seed"]
                    for model_instance in hp_groups[group]
                ]
            }
            for _class in classes:
                hp_score[group][f"f1-score, {_class}"] = [
                    model_instance[f"{_class}"]["f1-score"]
                    for model_instance in hp_groups[group]
                ]
                hp_score[group][f"precision, {_class}"] = [
                    model_instance[f"{_class}"]["precision"]
                    for model_instance in hp_groups[group]
                ]
                hp_score[group][f"recall, {_class}"] = [
                    model_instance[f"{_class}"]["recall"]
                    for model_instance in hp_groups[group]
                ]
        return hp_score

    def difference_hyperparameters(self, hp_groups):
        base_hps = hp_groups["base"][0]['hyperparameters']
        group_differences = {}
        for group in hp_groups:
            if group == "base":
                continue
            group_differences[group] = {}
            tmp_hps = hp_groups[group][0]["hyperparameters"]
            for hp in tmp_hps:
                if isinstance(tmp_hps[hp], int) or isinstance(tmp_hps, float):
                    if not np.allclose(tmp_hps[hp], base_hps[hp]):
                        group_differences[group][hp] = tmp_hps[hp]
                else:
                    if tmp_hps[hp] != base_hps[hp]:
                        group_differences[group][hp] = tmp_hps[hp]
        return group_differences

    # covariance strength across groups, as a measure of hyperparameter sensitivity?

    # so if f1_scores in base covary with f1_scores in hyperparameter then we are insensitive.
    # the stronger the similarity the more insensitive the hyperparameters.
    # we can also use ks-2sample test.
    # we can also use correlation
    # we can also use spearman correlation.
    # we can do a simple frequency count of the number of times the hyperparameters are uncorrelated/not similar
    # that gives us a coarse sense of sensitivity.  
    # we can then do a sum of strength of correlation when statistically significant. for something more granular.
    # we can also do a model version of this.  We regress f1_scores in A against f1_scores in B.  The better A 'fits' B, 
    # the less sensitive our thing is.
    # We can order the samples by seed.  Phew!        

    def sort_by_seed(self, hp_groups):
        new_hp_groups = {}
        for group in hp_groups:
            new_hp_groups[group] = sorted(
                hp_groups[group], 
                key=lambda t: t["seed"]
            )
        return new_hp_groups

    # the better the model does, the less sensitive the hyperparameter is
    # consider SVMR, DecisionTreeRegressor as well.  It's about learning
    # an inductive bias, not playing games to get a good fit.
    def get_model_relationship(
            self,
            model=LinearRegression()
    ):
        model_groups = pd.DataFrame()
        for group in self.hp_groups:
            if group == "base":
                continue
            for measure in self.score_groups[group]:
                y = np.array(self.score_groups["base"][measure])
                x = np.array(self.score_groups[group][measure]).reshape(-1, 1)
                X_train, X_test, y_train, y_test = train_test_split(
                    x, y, test_size=0.1
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                loss = mean_absolute_error(y_test, y_pred)
                for hp in self.group_diffs[group]:
                    model_groups = model_groups.append({
                        "group": group,
                        "measure": measure,
                        "loss": loss,
                        "hp_name": hp,
                        "hp_value": self.group_diffs[group][hp]
                    }, ignore_index=True)
        return model_groups

    # the stronger the correlation, the less sensitive the hyperparameter is
    def get_correlations(self,
        corr_func=stats.spearmanr
    ):
        """
        alt corr_func : stats.pearsonr or
        anything that implements correlation and pvalue
        as:
        res = corr_func(series_a, series_b)
        print(res.correlation, res.pvalue)
        """
        corr_groups = pd.DataFrame()
        for group in self.hp_groups:
            if group == "base":
                continue
            for measure in self.score_groups[group]:
                if measure == "seed":
                    continue
                corr = corr_func(
                    self.score_groups[group][measure],
                    self.score_groups["base"][measure]
                )
                for hp in self.group_diffs[group]:
                    corr_groups = corr_groups.append({
                        "group": group,
                        "measure": measure,
                        "correlation": corr.correlation,
                        "pvalue": corr.pvalue,
                        "hp_name": hp,
                        "hp_value": self.group_diffs[group][hp]
                    }, ignore_index=True)
        return corr_groups

    # fix start
    def get_variances(self):
        """
        """
        var_groups = pd.DataFrame()
        base_measures = {}
        groups = list(self.hp_groups.keys())
        for measure in self.score_groups[groups[0]]:
            if measure == "seed":
                continue
            base_measures[measure] = np.var(self.score_groups["base"][measure])
        for group in self.hp_groups:
            if group == "base":
                continue
            for measure in self.score_groups[group]:
                if measure == "seed":
                    continue
                var_diff = abs(
                    base_measures[measure] - np.var(self.score_groups[group][measure])
                )
                for hp in self.group_diffs[group]:
                    var_groups = var_groups.append({
                        "group": group,
                        "measure": measure,
                        "variance_diff": var_diff,
                        "hp_name": hp,
                        "hp_value": self.group_diffs[group][hp]
                    }, ignore_index=True)
        return var_groups

    def calc_concentration(self, series):
        """
        calculate density per decile
        """
        splits = [
            0, 0.1, 0.2, 0.3,
            0.4, 0.5, 0.6,
            0.7, 0.8, 0.9, 1
        ]
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        concentration_score = 0
        for index, split in enumerate(splits[1:]):
            concentration_score += len(series[
                (series > splits[index]) &
                (series < split)
            ]) * split
        return concentration_score

    def get_concentration(self):
        """
        """
        concentration_groups = pd.DataFrame()
        base_measures = {}
        groups = list(self.hp_groups.keys())
        for measure in self.score_groups[groups[0]]:
            if measure == "seed":
                continue
            base_measures[measure] = self.calc_concentration(
                self.score_groups["base"][measure]
            )
        for group in self.hp_groups:
            if group == "base":
                continue
            for measure in self.score_groups[group]:
                if measure == "seed":
                    continue
                other_measure = self.score_groups[group][measure]
                concentration_diff = abs(
                    base_measures[measure] - self.calc_concentration(other_measure)
                )
                for hp in self.group_diffs[group]:
                    concentration_groups = concentration_groups.append({
                        "group": group,
                        "measure": measure,
                        "concentration_diff": concentration_diff,
                        "hp_name": hp,
                        "hp_value": self.group_diffs[group][hp]
                    }, ignore_index=True)                    
        return concentration_groups

    def get_iqrs(self):
        """
        """
        iqr_groups = pd.DataFrame()
        base_measures = {}
        groups = list(self.hp_groups.keys())
        for measure in self.score_groups[groups[0]]:
            if measure == "seed":
                continue
            base_measures[measure] = stats.iqr(
                self.score_groups["base"][measure]
            )
        for group in self.hp_groups:
            if group == "base":
                continue
            for measure in self.score_groups[group]:
                if measure == "seed":
                    continue
                other_measure = self.score_groups[group][measure]
                iqr_diff = abs(
                    base_measures[measure] - stats.iqr(other_measure)
                )
                for hp in self.group_diffs[group]:
                    iqr_groups = iqr_groups.append({
                        "group": group,
                        "measure": measure,
                        "iqr_diff": iqr_diff,
                        "hp_name": hp,
                        "hp_value": self.group_diffs[group][hp]
                    }, ignore_index=True)
        return iqr_groups

    # unclear if smoothing will help
    # comparing each decile may still be more informative
    # run these tests against density smoothed data 
    # and original data
    #stats.cramervonmises_2samp
    #stats.ks_2samp

    def get_range_groups(self):
        """
        """
        range_groups = pd.DataFrame()
        base_measures = {}
        groups = list(self.hp_groups.keys())
        for measure in self.score_groups[groups[0]]:
            if measure == "seed":
                continue
            base_max = np.max(
                self.score_groups["base"][measure]
            )
            base_min = np.min(
                self.score_groups["base"][measure]
            )
            base_measures[measure] = base_max - base_min 
        for group in self.hp_groups:
            if group == "base":
                continue
            for measure in self.score_groups[group]:
                if measure == "seed":
                    continue
                _max = np.max(self.score_groups[group][measure])
                _min = np.min(self.score_groups[group][measure])
                _range = _max - _min
                _range_diff = abs(base_measures[measure] - _range)
                # base is missing from the gorup_diffs

                for hp in self.group_diffs[group]:
                    range_groups = range_groups.append({
                        "group": group,
                        "measure": measure,
                        "range_diff": _range_diff,
                        "hp_name": hp,
                        "hp_value": self.group_diffs[group][hp]
                    }, ignore_index=True)
        return range_groups

    def get_density(self):
        density_groups = pd.DataFrame()
        for group in self.hp_groups:
            for measure in self.score_groups[group]:
                if measure == "seed":
                    continue
                X = np.array(self.score_groups[group][measure]).reshape(-1, 1)
                kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
                density = kde.score_samples(X)
                if group == "base":
                    density_groups = density_groups.append({
                        "group": group,
                        "measure": measure,
                        "density": density,
                        "hp_name": "None",
                        "hp_value": "None"
                    }, ignore_index=True)
                else:
                    for hp in self.group_diffs[group]:
                        density_groups = density_groups.append({
                            "group": group,
                            "measure": measure,
                            "density": density,
                            "hp_name": hp,
                            "hp_value": self.group_diffs[group][hp]
                        }, ignore_index=True)
        return density_groups

    # this could be easier, just do inverse of correlation and add this to each value
    # this gives us a better sense of correlation in a less coarse way.

    def type_cast(self, value):
        if isinstance(value, np.float64):
            return float(value)
        elif isinstance(value, np.float32):
            return float(value)
        elif isinstance(value, np.float16):
            return float(value)
        else:
            return value
        
    def coarse_analyze_variance_sensitivity(self):
        num_groups = len(self.hp_groups.keys())
        base_hyperparameters = self.hp_groups["base"][0]["hyperparameters"]
        hp_sensitivity = self.var_groups.copy()
        hp_sensitivity["hp_sensitivity"] = 0
        for _, tmp in hp_sensitivity.groupby(by=["hp_name", "hp_value", "measure"]):
            indices = tmp[tmp["variance_diff"] > 10].index
            if indices.empty:
                continue
            else:
                hp_sensitivity.loc[indices, "hp_sensitivity"] = len(indices)        
        return hp_sensitivity

    def coarse_analyze_concentration_sensitivity(self):
        num_groups = len(self.hp_groups.keys())
        base_hyperparameters = self.hp_groups["base"][0]["hyperparameters"]
        hp_sensitivity = self.concentration_groups.copy()
        hp_sensitivity["hp_sensitivity"] = 0
        for _, tmp in hp_sensitivity.groupby(by=["hp_name", "hp_value", "measure"]):
            # I don't think 10 is right here.  Do some experiments.
            # this is a fine heuristic for now.
            indices = tmp[tmp["concentration_diff"] > 10].index
            if indices.empty:
                continue
            else:
                hp_sensitivity.loc[indices, "hp_sensitivity"] = len(indices)        
        return hp_sensitivity

    def coarse_analyze_iqr_sensitivity(self):
        num_groups = len(self.hp_groups.keys())
        base_hyperparameters = self.hp_groups["base"][0]["hyperparameters"]
        hp_sensitivity = self.iqr_groups.copy()
        hp_sensitivity["hp_sensitivity"] = 0
        for _, tmp in hp_sensitivity.groupby(by=["hp_name", "hp_value", "measure"]):
            indices = tmp[tmp["iqr_diff"] > 5].index
            if indices.empty:
                continue
            else:
                hp_sensitivity.loc[indices, "hp_sensitivity"] = len(indices)        
        return hp_sensitivity

    def coarse_analyze_range_sensitivity(self):
        num_groups = len(self.hp_groups.keys())
        base_hyperparameters = self.hp_groups["base"][0]["hyperparameters"]
        hp_sensitivity = self.range_groups.copy()
        hp_sensitivity["hp_sensitivity"] = 0
        for _, tmp in hp_sensitivity.groupby(by=["hp_name", "hp_value", "measure"]):
            indices = tmp[tmp["range_diff"] > 15].index
            if indices.empty:
                continue
            else:
                hp_sensitivity.loc[indices, "hp_sensitivity"] = len(indices)        
        return hp_sensitivity

    def coarse_analyze_density_sensitivity_ks_2samp(self):
        num_groups = len(self.hp_groups.keys())
        base_hyperparameters = self.hp_groups["base"][0]["hyperparameters"]
        hp_sensitivity = self.density_groups.copy()
        hp_sensitivity["marginal_hp_sensitivity"] = 0
        hp_sensitivity["hp_sensitivity"] = 0
        for (_,measure,_,_), tmp in hp_sensitivity.groupby(
                by=["group", "measure", "hp_name", "hp_value"]
        ):
            base = hp_sensitivity[
                (hp_sensitivity["group"] == "base") &
                (hp_sensitivity["measure"] == measure)
            ]
            # there should only be one row
            res = stats.ks_2samp(
                base["density"].iloc[0],
                tmp["density"].iloc[0]
            )
            if res.pvalue < 0.05:
                hp_sensitivity.loc[tmp.index, "marginal_hp_sensitivity"] = 1
        for (measure, hp_name, hp_value), tmp in hp_sensitivity.groupby(
                by=["measure", "hp_name", "hp_value"]
        ):
            hp_sensitivity.loc[
                tmp.index, "hp_sensitivity"
            ] = tmp["marginal_hp_sensitivity"].sum()
        return hp_sensitivity

    def coarse_analyze_density_sensitivity_cramervonmises_2samp(self):
        num_groups = len(self.hp_groups.keys())
        base_hyperparameters = self.hp_groups["base"][0]["hyperparameters"]
        hp_sensitivity = self.density_groups.copy()
        hp_sensitivity["marginal_hp_sensitivity"] = 0
        hp_sensitivity["hp_sensitivity"] = 0
        for (_,measure,_,_), tmp in hp_sensitivity.groupby(
                by=["group", "measure", "hp_name", "hp_value"]
        ):
            base = hp_sensitivity[
                (hp_sensitivity["group"] == "base") &
                (hp_sensitivity["measure"] == measure)
            ]
            # there should only be one row
            res = stats.cramervonmises_2samp(
                base["density"].iloc[0],
                tmp["density"].iloc[0]
            )
            if res.pvalue < 0.05:
                hp_sensitivity.loc[tmp.index, "marginal_hp_sensitivity"] = 1
        for (measure, hp_name, hp_value), tmp in hp_sensitivity.groupby(
                by=["measure", "hp_name", "hp_value"]
        ):
            hp_sensitivity.loc[
                tmp.index, "hp_sensitivity"
            ] = tmp["marginal_hp_sensitivity"].sum()
        return hp_sensitivity
    
    def coarse_analyze_correlation_sensitivity(self):
        num_groups = len(self.hp_groups.keys())
        base_hyperparameters = self.hp_groups["base"][0]["hyperparameters"]
        hp_sensitivity = self.corr_groups.copy()
        hp_sensitivity["hp_sensitivity"] = 0
        for _, tmp in hp_sensitivity.groupby(by=["hp_name", "hp_value", "measure"]):
            indices = tmp[(
                tmp["correlation"] < 0.2
            ) & (
                tmp["pvalue"] < 0.07
            )].index
            if indices.empty:
                continue
            else:
                hp_sensitivity.loc[indices, "hp_sensitivity"] = len(indices)
        return hp_sensitivity

    def coarse_analyze_model_based_sensitivity(self):
        num_groups = len(self.hp_groups.keys())
        base_hyperparameters = self.hp_groups["base"][0]["hyperparameters"]
        hp_sensitivity = self.model_groups.copy()
        hp_sensitivity["hp_sensitivity"] = 0
        for _, tmp in hp_sensitivity.groupby(by=["hp_name", "hp_value", "measure"]):

            indices = tmp[tmp["loss"] < 0.5].index
            if indices.empty:
                continue
            else:
                hp_sensitivity.loc[indices, "hp_sensitivity"] = len(indices)
        return hp_sensitivity

    def fine_analyze_correlation_sensitivity(self):
        """
        Bigger values imply greater sensitivity, in absolute terms
        the more experiments you run the bigger this number will be.
        """
        num_groups = len(self.hp_groups.keys())
        base_hyperparameters = self.hp_groups["base"][0]["hyperparameters"]
        hp_sensitivity = self.corr_groups.copy()
        hp_sensitivity["hp_sensitivity"] = 0
        for _, tmp in hp_sensitivity.groupby(by=["hp_name", "hp_value", "measure"]):
            indices = tmp.index
            if indices.empty:
                continue
            else:
                hp_sensitivity.loc[indices, "hp_sensitivity"] = 1/(
                    tmp["correlation"].sum() + tmp["pvalue"].sum()
                )
        return hp_sensitivity

    def fine_analyze_variance_sensitivity(self):
        """
        Bigger values imply greater sensitivity, in absolute terms
        the more experiments you run the bigger this number will be.
        """
        num_groups = len(self.hp_groups.keys())
        base_hyperparameters = self.hp_groups["base"][0]["hyperparameters"]
        hp_sensitivity = self.var_groups.copy()
        hp_sensitivity["hp_sensitivity"] = 0
        for _, tmp in hp_sensitivity.groupby(by=["hp_name", "hp_value", "measure"]):
            indices = tmp.index
            if indices.empty:
                continue
            else:
                hp_sensitivity.loc[indices, "hp_sensitivity"] = tmp["variance_diff"].sum()
        return hp_sensitivity

    def fine_analyze_concentration_sensitivity(self):
        """
        Bigger values imply greater sensitivity, in absolute terms
        the more experiments you run the bigger this number will be.
        """
        num_groups = len(self.hp_groups.keys())
        base_hyperparameters = self.hp_groups["base"][0]["hyperparameters"]
        hp_sensitivity = self.concentration_groups.copy()
        hp_sensitivity["hp_sensitivity"] = 0
        for _, tmp in hp_sensitivity.groupby(by=["hp_name", "hp_value", "measure"]):
            indices = tmp.index
            if indices.empty:
                continue
            else:
                hp_sensitivity.loc[indices, "hp_sensitivity"] = tmp["concentration_diff"].sum()
        return hp_sensitivity        

    def fine_analyze_iqr_sensitivity(self):
        """
        Bigger values imply greater sensitivity, in absolute terms
        the more experiments you run the bigger this number will be.
        """
        num_groups = len(self.hp_groups.keys())
        base_hyperparameters = self.hp_groups["base"][0]["hyperparameters"]
        hp_sensitivity = self.iqr_groups.copy()
        hp_sensitivity["hp_sensitivity"] = 0
        for _, tmp in hp_sensitivity.groupby(by=["hp_name", "hp_value", "measure"]):
            indices = tmp.index
            if indices.empty:
                continue
            else:
                hp_sensitivity.loc[indices, "hp_sensitivity"] = tmp["iqr_diff"].sum()
        return hp_sensitivity        

    def fine_analyze_range_sensitivity(self):
        """
        Bigger values imply greater sensitivity, in absolute terms
        the more experiments you run the bigger this number will be.
        """
        num_groups = len(self.hp_groups.keys())
        base_hyperparameters = self.hp_groups["base"][0]["hyperparameters"]
        hp_sensitivity = self.range_groups.copy()
        hp_sensitivity["hp_sensitivity"] = 0
        for _, tmp in hp_sensitivity.groupby(by=["hp_name", "hp_value", "measure"]):
            indices = tmp.index
            if indices.empty:
                continue
            else:
                hp_sensitivity.loc[indices, "hp_sensitivity"] = tmp["range_diff"].sum()
        return hp_sensitivity        

    def fine_analyze_model_based_sensitivity(self):
        """
        Bigger values imply greater sensitivity, in absolute terms
        the more experiments you run the bigger this number will be.
        """
        num_groups = len(self.hp_groups.keys())
        base_hyperparameters = self.hp_groups["base"][0]["hyperparameters"]
        hp_sensitivity = self.model_groups.copy()
        hp_sensitivity["hp_sensitivity"] = 0
        for _, tmp in hp_sensitivity.groupby(by=["hp_name", "hp_value", "measure"]):
            indices = tmp.index
            if indices.empty:
                continue
            else:
                hp_sensitivity.loc[indices, "hp_sensitivity"] = tmp["loss"].sum()
        return hp_sensitivity        

    def fine_analyze_density_sensitivity_ks_2samp(self):
        """
        Bigger values imply greater sensitivity, in absolute terms
        the more experiments you run the bigger this number will be.
        """
        num_groups = len(self.hp_groups.keys())
        base_hyperparameters = self.hp_groups["base"][0]["hyperparameters"]
        hp_sensitivity = self.density_groups.copy()
        hp_sensitivity["marginal_hp_sensitivity"] = 0
        hp_sensitivity["hp_sensitivity"] = 0
        for (_,measure,_,_), tmp in hp_sensitivity.groupby(
                by=["group", "measure", "hp_name", "hp_value"]
        ):
            base = hp_sensitivity[
                (hp_sensitivity["group"] == "base") &
                (hp_sensitivity["measure"] == measure)
            ]
            # there should only be one row
            res = stats.ks_2samp(
                base["density"].iloc[0],
                tmp["density"].iloc[0]
            )
            hp_sensitivity.loc[tmp.index, "marginal_hp_sensitivity"] = res.statistic
        for (measure, hp_name, hp_value), tmp in hp_sensitivity.groupby(
                by=["measure", "hp_name", "hp_value"]
        ):
            hp_sensitivity.loc[
                tmp.index, "hp_sensitivity"
            ] = tmp["marginal_hp_sensitivity"].sum()
        return hp_sensitivity

    def fine_analyze_density_sensitivity_cramervonmises_2samp(self):
        """
        Bigger values imply greater sensitivity, in absolute terms
        the more experiments you run the bigger this number will be.
        """
        num_groups = len(self.hp_groups.keys())
        base_hyperparameters = self.hp_groups["base"][0]["hyperparameters"]
        hp_sensitivity = self.density_groups.copy()
        hp_sensitivity["marginal_hp_sensitivity"] = 0
        hp_sensitivity["hp_sensitivity"] = 0
        for (_,measure,_,_), tmp in hp_sensitivity.groupby(
                by=["group", "measure", "hp_name", "hp_value"]
        ):
            base = hp_sensitivity[
                (hp_sensitivity["group"] == "base") &
                (hp_sensitivity["measure"] == measure)
            ]
            # there should only be one row
            res = stats.cramervonmises_2samp(
                base["density"].iloc[0],
                tmp["density"].iloc[0]
            )
            hp_sensitivity.loc[tmp.index, "marginal_hp_sensitivity"] = res.statistic
        for (measure, hp_name, hp_value), tmp in hp_sensitivity.groupby(
                by=["measure", "hp_name", "hp_value"]
        ):
            hp_sensitivity.loc[
                tmp.index, "hp_sensitivity"
            ] = tmp["marginal_hp_sensitivity"].sum()
        return hp_sensitivity

