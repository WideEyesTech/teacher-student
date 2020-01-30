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

if __name__ == "__main__":
    # Load config
    CONFIG = json.load(open("./config/CenterNet104_teacher_student.json"))
    db = WeakLabels(CONFIG)
    # Load model
    MODEL = nnet(db)
    # Load snapshot
    SNAPSHOT = torch.load("./cache/nnet/CenterNet-104_480000.pkl", map_location="cuda:0")
    
    # Weights had been saved with DataParallel so we delete the "module." prefix to load them with single GPU
    WEIGHTS = OrderedDict()
    for k, v in SNAPSHOT.items():
        WEIGHTS[k.replace('module.', '')] = v

    # Load weights into the module
    MODEL.load_state_dict(WEIGHTS)

    if torch.cuda.is_available():
        MODEL.cuda()
    
    # Set evaluation mode
    MODEL.eval()

    # Dir to save results
    result_dir = system_configs.result_dir

    testing(db, MODEL, result_dir)