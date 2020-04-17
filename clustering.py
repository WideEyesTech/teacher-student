import cytools

from os.path import join as pjoin
from os.path import exists as pexists
from os.path import split as psplit
import json
import random
import csv
import time
from random import shuffle
import tqdm

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Enum like dict wih indices for each classname
classes = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "airplane": 5,
    "bus": 6,
    "train": 7,
    "truck": 8,
    "boat": 9,
    "trafficlight": 10,
    "firehydrant": 11,
    "stopsign": 13,
    "parkingmeter": 14,
    "bench": 15,
    "bird": 16,
    "cat": 17,
    "dog": 18,
    "horse": 19,
    "sheep": 20,
    "cow": 21,
    "elephant": 22,
    "bear": 23,
    "zebra": 24,
    "giraffe": 25,
    "backpack": 27,
    "umbrella": 28,
    "handbag": 31,
    "tie": 32,
    "suitcase": 33,
    "frisbee": 34,
    "skis": 35,
    "snowboard": 36,
    "sportsball": 37,
    "kite": 38,
    "baseballbat": 39,
    "baseballglove": 40,
    "skateboard": 41,
    "surfboard": 42,
    "tennisracket": 43,
    "bottle": 44,
    "wineglass": 46,
    "cup": 47,
    "fork": 48,
    "knife": 49,
    "spoon": 50,
    "bowl": 51,
    "banana": 52,
    "apple": 53,
    "sandwich": 54,
    "orange": 55,
    "broccoli": 56,
    "carrot": 57,
    "hotdog": 58,
    "pizza": 59,
    "donut": 60,
    "cake": 61,
    "chair": 62,
    "couch": 63,
    "pottedplant": 64,
    "bed": 65,
    "diningtable": 67,
    "toilet": 70,
    "tv": 72,
    "laptop": 73,
    "mouse": 74,
    "remote": 75,
    "keyboard": 76,
    "cellphone": 77,
    "microwave": 78,
    "oven": 79,
    "toaster": 80,
    "sink": 81,
    "refrigerator": 82,
    "book": 84,
    "clock": 85,
    "vase": 86,
    "scissors": 87,
    "teddybear": 88,
    "hairdrier": 89,
    "toothbrush": 90,
}


def print_results(results, model, image):
    fig, ax = plt.subplots(figsize=(12, 12))
    fig = ax.imshow(image, aspect='equal')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    for result in results:

        bbox = result[:4]
        score = result[4:-1].max()

        # Get label
        label = list(classes.keys())[list(classes.values()).index(
            int(np.array(result[4:-1]).argmax()))]

        print("Teacher: ", model)
        print("Score: ", score)
        print("Category: ", label)
        print("Number of teacher inferences: ", result[-1])

        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]

        opacity = 1
        if model == "cluster":
            opacity = (1-(1/result[-1]))**2

        if float("{0:.1f}".format(score)) < 0:
            score_color = 0
        else:
            score_color = (
                1-(float("{0:.1f}".format(score)))**3, .6, .0, opacity)

        ax.set_title(model)

        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, edgecolor=score_color, linewidth=4.0))
        ax.text(xmin+1, ymin-3, '{:s}'.format("{}_{}".format(label, score)), bbox=dict(
            facecolor=score_color, ec='black', lw=2, alpha=0.5), fontsize=15, color='white', weight='bold')
    import pdb; pdb.set_trace()
    plt.show()
    plt.close()


def merge_scores(bboxA, bboxB, method="mean"):
    if method == "mean":
        return (bboxA+bboxB)/2


def merge_bboxes(bboxA, bboxB, method="mean"):
    if method == "mean":
        return (bboxA+bboxB)/2


# bboxes -> [x1, y1, x2, y2, score1, score2, ..., scoreN, category]
def combine(bboxesA, bboxesB, thr=.5):
    ious = cytools.bbox_overlaps(bboxesA[:, :4], bboxesB[:, :4])
    # [
    #   [0.4, 0.0, 0.05, 0.1, 0.9],
    #   [0.9, 0.23, 0.0, 0.1, 0.1],
    #   [0.6, 0.1, 0.2, 0.1, 0.1],
    # ]

    max_ious = ious.max(1)  # [0.9, 0.9, 0.6]

    oks = max_ious > thr
    # [True, True, False]

    rels = ious.argmax(1)
    # [4, 0, 0]

    if oks.size:
        bboxesC = np.zeros(bboxesA[oks, :].shape)
        bboxesC[:, :-1] = (bboxesA[oks, :-1] + bboxesB[rels[oks], :-1]) / 2
        bboxesC[:, -1] = (bboxesA[oks, -1] + bboxesB[rels[oks], -1])

    no_oks = oks == False
    # [False, False, True]

    bboxesC = np.vstack((bboxesC, bboxesA[no_oks, :]))

    max_iousB = ious.max(0)  # [0.9, 0.23, 0.2, 0.1, 0.9]

    oksB = max_iousB > thr
    # [True, False, False, False, True]

    no_oksB = oksB == False
    # [False, True, True, True, False]

    bboxesC = np.vstack((bboxesC, bboxesB[no_oksB, :]))

    # [x1, y1, x2, y2, score1, score2, ..., scoreN, category]
    test = [x[4:-1].argmax() for x in bboxesC]

    return bboxesC


def parse_teacher_results(inference):
    # Parse category name to make data more consistent removing underscores and spaces
    category = inference["category_id"].replace("_", "").replace(" ", "")

    # Transform score to a len(classes) 0 valued list
    # with the score value in the category position of the list
    # [0, 0, 0 ,0 , 0.8, ... 0]
    # 91 for 90 posible classes index and 1 for the number of teachers inferences
    score_and_n_of_teachers = [0]*(91+1)
    # Minimum one techer have inference
    score_and_n_of_teachers[-1] = 1.0

    category_position = classes[category]
    score_and_n_of_teachers[category_position] = inference["score"]

    # Return new data structure
    return inference["bbox"] + score_and_n_of_teachers


# Convert bbox from: xmin, ymin, width, height -- to --> xmin, ymin, xmax, ymax
def convert_bbox(inference):

    # We cannot mutate original iterator

    x1 = inference["bbox"][0]
    x2 = inference["bbox"][2] + inference["bbox"][0]
    y1 = inference["bbox"][1]
    y2 = inference["bbox"][3] + inference["bbox"][1]

    return {
        "category_id": inference["category_id"],
        "bbox": [x1, y1, x2, y2],
        "score": inference["score"]
    }


def save_result(input_path, output_path, annotations, file):
    # Paths
    images_path = "{}_images.csv".format(output_path)
    ann_paths = "{}_ann.csv".format(output_path)

    # Get image HxW
    img = cv2.imread(input_path)
    dimensions = img.shape

    # Get an ID
    a_id = int(str(time.time()).replace(".", ""))

    ref, _ = psplit(file)

    with open(images_path, 'a') as f:
        wr = csv.writer(f)

        wr.writerow((dimensions[1], dimensions[0], "{}.jpg".format(
            ref), "http://{}.jpg".format(ref), ref))

    with open(ann_paths, 'a') as f:
        wr = csv.writer(f)

        # Add annotation
        for annotation in annotations:
            wr.writerow((0, ref, annotation[:4].tolist(), int(
                annotation[4:-1].argmax()), a_id, int(annotation[4:-1].max())))


def cluster():

    # Enable debugging features
    debug = True

    # Make sure the experiment does not repeat
    random.seed(int(time.time()))
    # random.seed(0)

    # Paths
    data_dir = "/home/toni/datasets/openimages"
    filenames_paths = [x.strip() for x in tqdm.tqdm(open(
        "/home/toni/datasets/sorted_4_oi_names.txt"), desc="Reading filenames")]
    shuffle(filenames_paths)
    inferences_jsons = [x.strip()[:-4] + "/results.json" for x in tqdm.tqdm(
        filenames_paths, desc="Creating reasults paths")]
    inferences_path = "/home/toni/datasets/results"
    results_path = "/home/toni/datasets/openimages/annotations"

    teachers = [
        "CenterNet-104_480000",
        "ATSS",
        "GCNET"
    ]

    # Clustering results
    cluster_result = []
     
    counter = 0
    # Create csv columnssudo apt install nfs-common
    for count in tqdm.tqdm(range(0, len(inferences_jsons)), ncols=80, desc="Clustering..."):
        if counter ==  100000:
            break
        file = inferences_jsons[count]

        # Check if all teachers have inferences
        # of the file, otherwise skip loop
        all_teachers_have_inferred = True
        for teacher in teachers:
            # Get results path for each teacher
            teacher_inferences = pjoin(inferences_path, teacher)

            if not pexists(pjoin(teacher_inferences, file)):
                all_teachers_have_inferred = False
                break

        if not all_teachers_have_inferred:
            if debug:
                print("Skipping {}...".format(count))
            continue

        # Get image
        if debug:
            print(file)
            image_file = pjoin(data_dir, filenames_paths[count])
            image = cv2.imread(image_file)[:, :, ::-1]

        # In case all teachers have inferences
        # of the file

        # Make sure result is
        # ignored if not all json
        # files load correct
        jsonLoadFailed = False
 
        # Start clustering
        for index, teacher in enumerate(teachers):

            if teacher != "GCNET":
                continue

            # Get results path for each teacher
            teacher_inferences = pjoin(inferences_path, teacher)
            # Get results path for each FILE
            file_inferences = pjoin(teacher_inferences, file)

            # Read teacher file inferences
            with open(file_inferences) as f_i:
                try:
                    inferences = json.load(f_i)
                except:
                    print("failed to load: ", file_inferences)
                    jsonLoadFailed = True
                    break

            inferences = np.array(inferences)

            # We need to parse Centernet bbox, so we must check
            # if teacher is centernet
            if teacher == "CenterNet-104_480000":
                inferences = list(map(convert_bbox, inferences))

            # Filter inferences by score
            if teacher == "GCNET":
                inferences = np.array(list(filter(lambda x: x["score"] > 0.6, inferences)))
                if len(inferences) == 0:
                    continue

            # Transform results data structure from dict to list
            # in order to speed up clustering between inferences
            inferences = np.array(list(map(parse_teacher_results, inferences)))

            if debug:
                print_results(inferences, teacher, image)

            # Do not cluster until there are at least two teachers
            #if index == 0:
            #    cluster_result = inferences
            #    continue
            cluster_result = inferences

            #cluster_result = combine(cluster_result, inferences)

        # If some json have failed to load skip clustering
        if jsonLoadFailed:
            continue

        # Print clustering result
        if debug:
            print_results(cluster_result, "cluster", image)

        # Split between train test and validation
        save_as_type = "train"
        r = random.random()

        save_as_type = "train"

        counter+=1
        
        #save_result(pjoin(
        #    data_dir, filenames_paths[count]), "{}/instances_{}2017".format(results_path, save_as_type), cluster_result, file)

        yield count


if __name__ == "__main__":
    # uncomment to test only one image
    # next(cluster())
    # return

    NUMBER_OF_EXAMPLES = 40000000

    for i, x in enumerate(cluster()):
        if i == NUMBER_OF_EXAMPLES:
            break
        continue
