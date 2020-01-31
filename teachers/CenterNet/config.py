import os
import numpy as np


class Config:
    def __init__(self):
        self._configs = {}
        self._configs["sampling_function"] = "kp_detection"

        # Directories
        self._configs["data_dir"] = "/home/toni/datasets/openimages"
        self._configs["filenames_dir"] = "/tmp/oi_names.txt"
        self._configs["cache_dir"] = "/home/toni/Desktop/teacher-student/teachers/CenterNet/cache"
        self._configs["snapshot_name"] = "CenterNet-104_480000"
        self._configs["result_dir"] = "/opt/results"
        self._configs["model_config"] = "/home/toni/Desktop/teacher-student/teachers/CenterNet/config/CenterNet104_teacher_student.json"


        # Rng
        self._configs["data_rng"] = np.random.RandomState(123)
        self._configs["nnet_rng"] = np.random.RandomState(317)


    @property
    def model_config(self):
        return self._configs["model_config"]

    @property
    def sampling_function(self):
        return self._configs["sampling_function"]

    @property
    def data_rng(self):
        return self._configs["data_rng"]

    @property
    def nnet_rng(self):
        return self._configs["nnet_rng"]

    @property
    def result_dir(self):
        result_dir = os.path.join(
            self._configs["result_dir"], self.snapshot_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        return result_dir

    @property
    def dataset(self):
        return self._configs["dataset"]

    @property
    def snapshot_name(self):
        return self._configs["snapshot_name"]

    @property
    def snapshot_dir(self):
        snapshot_dir = os.path.join(self.cache_dir, "nnet", self.snapshot_name)

        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)

        return snapshot_dir

    @property
    def snapshot_file(self):
        snapshot_file = os.path.join(
            self.snapshot_dir + ".pkl")
        return snapshot_file

    @property
    def config_dir(self):
        return self._configs["config_dir"]

    @property
    def batch_size(self):
        return self._configs["batch_size"]

    @property
    def max_iter(self):
        return self._configs["max_iter"]

    @property
    def data_dir(self):
        return self._configs["data_dir"]

    @property
    def filenames_dir(self):
        return self._configs["filenames_dir"]

    @property
    def cache_dir(self):
        if not os.path.exists(self._configs["cache_dir"]):
            os.makedirs(self._configs["cache_dir"])
        return self._configs["cache_dir"]

    def update_config(self, new):
        for key in new:
            if key in self._configs:
                self._configs[key] = new[key]


system_configs = Config()
