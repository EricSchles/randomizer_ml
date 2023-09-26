import os
import pickle

class Experiment:
    def __init__(self, _id, experiments_directory=None):
        if experiments_directory is None:
            self.experiments_directory = "experiments"
        else:
            self.experiments_directory = experiments_directory

        self.id = _id
        self.check_experiment_id()

    def check_experiment_id(self):
        experiment_path = self.get_experiment_path()
        if os.path.exists(experiment_path):
            raise Exception('''
            experiment id's must be unique.  Please choose a new name.
            ''')

    def get_experiment_path(self):
        return self.experiments_directory + "/" + self.id
    
    def __enter__(self):
        experiment_path = self.get_experiment_path()
        meta_data_path = experiment_path + "/meta_data.md"
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        if not os.path.exists(meta_data_path):
            with open(meta_data_path, "w") as f:
                f.write('')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def log_model_instances(self, model_instances):
        experiment_path = self.get_experiment_path()
        pickle.dump(
            model_instances,
            open(
                experiment_path+"/model_instances.pk",
                "wb"
            )
        )

    def log_model(self, model):
        experiment_path = self.get_experiment_path()
        meta_data = experiment_path + "/meta_data.md"
        with open(meta_data, "a") as f:
            f.write("model name: "+str(model.__class__))
            
    def log_num_trials(self, num_trials):
        experiment_path = self.get_experiment_path()
        meta_data = experiment_path + "/meta_data.md"
        with open(meta_data, "a") as f:
            f.write("num_trials: "+str(num_trials))

        

    
