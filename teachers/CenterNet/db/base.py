from os import getpid
from os.path import exists as pexists
from os.path import join as pjoin
from multiprocessing import Pool
import h5py
import numpy as np
from random import shuffle
import tqdm
from config import system_configs


def check_detection_file(params):
    filename = params
    # Check if a folder with same path exists (check if predictino exists)
    return pexists(filename)


class BASE(object):
    def __init__(self):

        # Get images ids
        self._image_ids = []

        params = []
        filenames = []
        with open(system_configs.filenames_dir, "r") as fid:
            for x in fid:
                x = x.strip()
                x_f = x.replace(".jpg", "")
                params.append(pjoin(system_configs.result_dir, x_f))
                filenames.append(x)

        pool = Pool(16)
        res = list(tqdm.tqdm(pool.imap(check_detection_file,
                                       params),
                             total=len(params)))
        # res = pool.map(check_detection_file, params)

        self._image_ids = [x for x, y in zip(filenames, res) if not y]
        
        shuffle(self._image_ids)
        
        self._db_inds = np.arange(len(self._image_ids))
        self._score_treshold = 0.7

        self._image_file      = system_configs.data_dir + "/{}"

        self._mean    = np.zeros((3, ), dtype=np.float32)
        self._std     = np.ones((3, ), dtype=np.float32)
        self._eig_val = np.ones((3, ), dtype=np.float32)
        self._eig_vec = np.zeros((3, 3), dtype=np.float32)

        self._configs             = {}
        self._train_cfg           = {}
        self._model               = {}
        self._configs["data_aug"] = True

        self._data_rng            = None

    @property
    def data(self):
        if self._data is None:
            raise ValueError("data is not set")
        return self._data

    @property
    def configs(self):
        return self._configs

    @property
    def score_treshold(self):
        return self._score_treshold

    @property
    def ids(self):
        return self._image_ids

    @property
    def train_cfg(self):
        return self._train_cfg

    @property
    def model(self):
        return self._model

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def eig_val(self):
        return self._eig_val

    @property
    def eig_vec(self):
        return self._eig_vec

    @property
    def db_inds(self):
        return self._db_inds

    @property
    def split(self):
        return self._split

    def update_config(self, new):
        for key in new:
            if key in self._configs:
                self._configs[key] = new[key]

    def image_ids(self, ind):
        return self._image_ids[ind]

    def image_file(self, ind):
        if self._image_file is None:
            raise ValueError("Image path is not initialized")

        image_id = self._image_ids[ind]
        return self._image_file.format(image_id)

    def write_result(self, ind, all_bboxes, all_scores):
        pass


    def shuffle_inds(self, quiet=False):
        if self._data_rng is None:
            self._data_rng = np.random.RandomState(os.getpid())

        if not quiet:
            print("shuffling indices...")
        rand_perm = self._data_rng.permutation(len(self._db_inds))
        self._db_inds = self._db_inds[rand_perm]
