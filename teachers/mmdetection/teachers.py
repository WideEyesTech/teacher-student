import argparse
import json
from os.path import join as pjoin
from os.path import exists as pexists
from random import shuffle

from multiprocessing import Pool
from pathlib import Path
import cv2
import torch
from tqdm import tqdm

from mmdet.apis import inference_detector, init_detector, show_result, LoadImage


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('data_dir')
    parser.add_argument('filenames_dir')
    parser.add_argument('results_dir')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    parser.add_argument('--debug', help="Save also images not just JSON")
    args = parser.parse_args()
    return args


def check_detection_file(params):
    filename = params
    # Check if a folder with same path exists (check if predictino exists)
    return pexists(filename)


class Dataset():

    def __init__(self, data_dir, filenames_dir, results_dir):

        # Paths
        self.data_dir = data_dir
        self.filenames_dir = filenames_dir
        self.results_dir = results_dir

        # Get images ids
        params = []
        filenames = []
        with open(self.filenames_dir, "r") as fid:
            for x in fid:
                x = x.strip()
                x_f = x.replace(".jpg", "")
                params.append(pjoin(self.results_dir, x_f))
                filenames.append(x)
        pool = Pool(16)
        res = list(tqdm(pool.imap(check_detection_file,
                                       params),
                             total=len(params)))
        self.image_ids = [x for x, y in zip(filenames, res) if not y]
        pool.close()
        shuffle(self.image_ids)
        # self.image_ids = [x.strip() for x in open(self.filenames_dir)]

    def getimage(self, idx):
        return cv2.imread(pjoin(self.data_dir, self.image_ids[idx]))


if __name__ == '__main__':
    args = parse_args()

    DEBUG = False
    DEMO = False

    dataset = Dataset(args.data_dir, args.filenames_dir, args.results_dir)

    for i in tqdm(range(len(dataset.image_ids)), ncols=80, desc="Predicting..."):

        # Paths
        result_path = args.results_dir + "/{}".format(image_id[:-4])
        Path(result_path).mkdir(parents=True, exist_ok=True)
        result_json = pjoin(result_path, "results.json")

        if pexists(result_json):
            continue

        # Get image
        image = dataset.getimage(i)
        image_id = dataset.image_ids[i]

        # Init model
        model = init_detector(
            args.config, args.checkpoint, device=torch.device('cuda', args.device))

        # Make inference
        result = inference_detector(model, image)

        detections = []

        for y, x in enumerate(result):
            for bbox in x:
                if bbox[-1] >= args.score_thr:
                    detections.append({
                        "image_id": y,
                        "category_id": model.CLASSES[y],
                        "bbox": [x.item() for x in bbox[:4]],
                        "score": bbox[-1].item()})

        if DEBUG:
            # Show results
            ir = show_result(
                image, result, model.CLASSES, args.score_thr, 1, False)
            cv2.imshow("im", ir)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if DEMO and i == 5:
            break


        # Save results
        if len(detections) != 0:
            with open(result_json, "w") as f:
                json.dump(detections, f)