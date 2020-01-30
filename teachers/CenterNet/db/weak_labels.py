""""""
""""""




from db.detection import DETECTION
class WeakLabels(DETECTION):
    """
    """

    def __init__(self, db_config):
        """
        """
        super(WeakLabels, self).__init__(db_config)
        # Load images

        self._cats = ['bg',
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
                      'toothbrush']

    def class_name(self, idx):
        return self._cats[idx]

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def parse_detections(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._cats[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

                    score = bbox[4]
                    bbox = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    if score > self.score_treshold:
                        detections.append(detection)

        return detections
