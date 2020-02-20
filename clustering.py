import cytools
from os.path import join as pjoin
from os.path import exists as pexists
import json
import random
import time
from random import shuffle

import cv2
import numpy as np
import matplotlib.pyplot as plt

CLASSES_BASIC = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush'
]


def print_results(results, model, image, color):
    fig, ax = plt.subplots(figsize=(12, 12))
    fig = ax.imshow(image, aspect='equal')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    for result in results:
        bbox = result[:4]

        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]

        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=color,
                                   linewidth=4.0))
        ax.text(xmin+1, ymin-3, '{:s}'.format(model+"_"+CLASSES_BASIC[int(result[-1])]), bbox=dict(facecolor=color, ec='black', lw=2, alpha=0.5),
                fontsize=15, color='white', weight='bold')

    plt.show()
    plt.close()


def merge_scores(bboxA, bboxB, method="mean"):
    if method == "mean":
        return (bboxA+bboxB)/2


def merge_bboxes(bboxA, bboxB, method="mean"):
    if method == "mean":
        return (bboxA+bboxB)/2


# bboxes -> [x1, y1, x2, y2, score1, score2, ..., scoreN, category]
def combine(bboxesA, bboxesB, thr=0.7):

    cluster = []

    ious = cytools.bbox_overlaps(bboxesA[:, :4], bboxesB[:, :4])
    # [
    #     [0.4, 0.0, 0.05, 0.1, 0.9],
    #     [0.9, 0.23, 0.0, 0.1, 0.1],
    #     [0.6, 0.1, 0.2, 0.1, 0.1],
    # ]

    max_ious = ious.max(1)  # [0.9, 0.9, 0.6]

    oks = max_ious > thr
    # [True, True, False]

    rels = ious.argmax(1)
    # [2, 4, 1, 3, 5]
    for i in range(len(oks)):
        if oks[i]:
            new_bbox = merge_bboxes(
                bboxesA[i, :4], bboxesB[rels[i], :4], method='mean')  # [x1, y1, x2, y2]
            # [score1, score2, ..., scoreN]
            new_score = merge_scores(
                bboxesA[i, 4:-1], bboxesB[rels[i], 4:-1], method='mean')
            cluster.append([*new_bbox, *new_score, bboxesA[i, -1]])
        else:
            cluster.append(bboxesA[i])

    max_iousB = ious.max(0)  # [0.9, 0.23, 0.2, 0.9]

    oksB = max_iousB > thr
    # [True, False, False, True]

    for x in range(len(bboxesB)):
        if not oksB[x]:
            cluster.append(bboxesB[x])

    # [x1, y1, x2, y2, score1, score2, ..., scoreN, category]
    return np.array(cluster)


def parse_teacher_results(x):
    # Enum like dict wih indices for each classname
    classes = {
        'airplane': 4,
        'apple': 47,
        'backpack': 24,
        'banana': 46,
        'baseballbat': 34,
        'baseballglove': 35,
        'bear': 21,
        'bed': 59,
        'bench': 13,
        'bicycle': 1,
        'bird': 14,
        'boat': 8,
        'book': 73,
        'bottle': 39,
        'bowl': 45,
        'broccoli': 50,
        'bus': 5,
        'cake': 55,
        'car': 2,
        'carrot': 51,
        'cat': 15,
        'cellphone': 67,
        'chair': 56,
        'clock': 74,
        'couch': 57,
        'cow': 19,
        'cup': 41,
        'diningtable': 60,
        'dog': 16,
        'donut': 54,
        'elephant': 20,
        'firehydrant': 10,
        'fork': 42,
        'frisbee': 29,
        'giraffe': 23,
        'hair_drier': 78,
        'handbag': 26,
        'horse': 17,
        'hotdog': 52,
        'keyboard': 66,
        'kite': 33,
        'knife': 43,
        'laptop': 63,
        'microwave': 68,
        'motorcycle': 3,
        'mouse': 64,
        'orange': 49,
        'oven': 69,
        'parkingmeter': 12,
        'person': 0,
        'pizza': 53,
        'pottedplant': 58,
        'refrigerator': 72,
        'remote': 65,
        'sandwich': 48,
        'scissors': 76,
        'sheep': 18,
        'sink': 71,
        'skateboard': 36,
        'skis': 30,
        'snowboard': 31,
        'spoon': 44,
        'sportsball': 32,
        'stopsign': 11,
        'suitcase': 28,
        'surfboard': 37,
        'teddybear': 77,
        'tennisracket': 38,
        'tie': 27,
        'toaster': 70,
        'toilet': 61,
        'toothbrush': 79,
        'trafficlight': 9,
        'train': 6,
        'truck': 7,
        'tv': 62,
        'umbrella': 25,
        'vase': 75,
        'wineglass': 40,
        'zebra': 22
    }

    # Parse category name to make data more consistent removing underscores and spaces
    category = x["category_id"].replace("_", "").replace(" ", "")

    # Transform score to a len(classes) 0 valued list
    # with the score value in the category position of the list
    # [0, 0, 0 ,0 , 0.8, ... 0]
    score = np.zeros(len(classes))
    category_position = classes[category]
    score[category_position] = x["score"]

    # Return new data structure
    return [*x["bbox"], *score, classes[category]]


# Convert bbox from: xmin, ymin, width, height -- to --> xmin, ymin, xmax, ymax
def convert_bbox(x):

    # We cannot mutate original iterator

    x1 = x["bbox"][0]
    x2 = x["bbox"][2] + x["bbox"][0]
    y1 = x["bbox"][1]
    y2 = x["bbox"][3] + x["bbox"][1]

    return {
        "category_id": x["category_id"],
        "bbox": [x1, y1, x2, y2],
        "score": x["score"]
    }


def cluster():

    # Enable debugging features
    debug = True

    # Make sure the experiment does not repeat
    random.seed(int(time.time()))

    # Paths
    data_dir = "/home/toni/datasets/openimages"
    filenames_paths = [x.strip() for x in open(
        "/home/toni/datasets/sorted_4_oi_names.txt")]
    shuffle(filenames_paths)
    results_paths = [x.strip()[:-4] + "/results.json" for x in filenames_paths]
    inferences_path = "/opt/results/"

    teachers = [
        "CenterNet-104_480000",
        "ATSS"
    ]

    # Clustering results
    cluster_result = []

    for count, file in enumerate(results_paths):
        # Get image
        if debug:
            image_file = pjoin(data_dir, filenames_paths[count])
            image = cv2.imread(image_file)[:, :, ::-1]

        # Check if all teachets have inferences
        # of the file, otherwise skip loop
        all_teachers_have_infered = True
        for teacher in teachers:
            # Get results path for each teacher
            teacher_inferences = pjoin(inferences_path, teacher)

            if not pexists(pjoin(teacher_inferences, file)):
                all_teachers_have_infered = False

        if not all_teachers_have_infered:
            continue

        # In case all teachers have inferences
        # of the file

        # Start clustering
        for i, teacher in enumerate(teachers):

            # Get results path for each teacher
            teacher_inferences = pjoin(inferences_path, teacher)
            # Get results path for each FILE
            file_inferences = pjoin(teacher_inferences, file)

            # Read teacher file inferences
            with open(file_inferences) as f_i:
                inferences = json.load(f_i)

            # Convert inferences to numpy
            inferences = np.array(inferences)

            # We need to parse Centernet bbox, so we must check
            # if teacher is centernet
            if teacher == "CenterNet-104_480000":
                inferences = list(map(convert_bbox, inferences))

            # Transform results data structure from dict to list
            # in order to speed up clustering between inferences
            inferences = list(map(parse_teacher_results, inferences))

            if debug:
                print_results(inferences, teacher, image, "red")

            # Do not cluster untill there are at least two teachers
            if i == 0:
                cluster_result = inferences
                continue

            cluster_result = combine(
                np.array(cluster_result), np.array(inferences))

        # Print clustering result
        if debug:
            print_results(cluster_result, "cluster", image, "green")

        yield count


if __name__ == "__main__":
    # uncomment to test only one image
    # next(cluster())
    # return

    number_of_examples = 1000

    for i, x in enumerate(cluster()):
        if i == number_of_examples:
            break
        continue
