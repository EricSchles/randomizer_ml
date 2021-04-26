import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, metrics, model_type):
        self.metrics = metrics
        self.model_type = model_type
        self.regression_metrics = [
            "mse", "max_error", "mae"
        ]
        if model_type == "classification":
            classes = list(metrics[0].keys())
            classes.remove("accuracy")
            classes.remove("mask")
            classes.remove("seed")
            classes.remove("hyperparameters")
            self.classes = classes
            self.classification_metrics = [
                "precision", "recall", "f1-score", "support"
            ]

    def visualize_regression(self, **kwargs):
        for metric in self.regression_metrics:
            metrics = [metrics[metric] for metrics in self.metrics]
            plt.hist(metrics, **kwargs)
            plt.xlabel(metric)
            plt.ylabel("magnitude")
            plt.show()

    def visualize_classification(self, **kwargs):
        for _class in self.classes:
            for metric in self.classification_metrics:
                metrics = [metrics[_class][metric] for metrics in self.metrics]
                plt.hist(metrics, **kwargs)
                plt.xlabel(metric)
                plt.ylabel("magnitude")
                if "avg" in _class:
                    plt.title(_class)
                else:
                    plt.title(f"class {_class}")
                plt.show()
