from collections import OrderedDict

import json
import numpy as np
import torch
import torchvision
from PIL import Image

from config import system_configs
from db.weak_labels import WeakLabels
from models.CenterNet104 import model as nnet
from test.openimages import testing

class GenerateWeakLabels():
    @staticmethod
    def generate():
        """
        Generate weak labels
        """
        # Load config
        CONFIG = json.load(open(system_configs.model_config))
        db = WeakLabels(CONFIG)
        # Load model
        MODEL = nnet(db)
        # Load snapshot
        SNAPSHOT = torch.load(system_configs.snapshot_file, map_location="cuda:0")
        
        # Weights had been saved with DataParallel so we delete the "module." prefix to load them with single GPU
        WEIGHTS = OrderedDict()
        for k, v in SNAPSHOT.items():
            WEIGHTS[k.replace('module.', '')] = v

        # Load weights into the module
        MODEL.load_state_dict(WEIGHTS)

        if torch.cuda.is_available():
            MODEL.cuda()
        import pdb; pdb.set_trace()
        # Set evaluation mode
        MODEL.eval()

        # Dir to save results
        result_dir = system_configs.result_dir

        testing(db, MODEL, result_dir)

if __name__ == "__main__":
    GenerateWeakLabels.generate()