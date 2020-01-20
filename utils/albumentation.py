from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, RandomRotate90,
    Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    RandomBrightnessContrast, OneOf, Compose, Resize, Flip, Transpose, CLAHE,
    IAAAdditiveGaussianNoise, IAASharpen, IAAEmboss
)

from albumentations.core.composition import KeypointParams, BboxParams

class Albumentation():
    """Albumentation module for image augmentation
    """

    def __init__(self, label_type, resize):
        """ Add docstring !!!
        """
        self.label_type = label_type
        self.resize = resize
        self.transforms = [
            HorizontalFlip(),
            ShiftScaleRotate(shift_limit=0.0625,
                             scale_limit=0.2, rotate_limit=45, p=0.2),
            RandomRotate90(),
            HueSaturationValue(p=0.3),
            OneOf([
                MotionBlur(p=0.2),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            Flip(),
            Transpose(),
            OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomBrightnessContrast(),
            ], p=0.3),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2),
            RandomRotate90(),
            Resize(*self.resize),
        ]

    def strong_aug(self, p=1):
        """ Add docstring !!!
        """
        if self.label_type == 'kp':
            return Compose(self.transforms, p=p, keypoint_params=KeypointParams(format='xy'))

        #if self.label_type == 'bb':
            #TODO return Compose(self.transforms, p=p, bbox_params=BboxParams(formar='coco'))

        # TODO should add defaul return
        return 0

    def fast_aug(self, p=1):
        """ Add docstring !!!
        """
        fast_transform = [
            HorizontalFlip(),
            OneOf([
                MotionBlur(p=0.2),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            Resize(*self.resize),
        ]
        if self.label_type == 'kp':
            return Compose(fast_transform, p=p, keypoint_params=KeypointParams(format='xy'))

        #if self.label_type == 'bb':
            # TODO return Compose(fast_transform, p=p, bbox_params=BboxParams(formar='coco'))

        # TODO should add default return
        return 0
