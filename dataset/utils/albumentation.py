from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, RandomRotate90,
    Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    RandomBrightnessContrast, OneOf, Compose, Resize, Flip, Transpose, CLAHE,
    IAAAdditiveGaussianNoise, IAASharpen, IAAEmboss, ElasticTransform, Flip, Rotate
)

from albumentations.core.composition import KeypointParams, BboxParams


class Albumentation():
    """Albumentation module for image augmentation
    """

    def __init__(self, label_type, params, transform_type, p=0.5):
        """ Add docstring !!!
        """
        self.label_type = label_type
        self.params = params
        self.transform_type = transform_type
        self.p = p
        self.transforms = {
            "pixel_level": OneOf([
                Blur(),
                GaussNoise(),
                IAAAdditiveGaussianNoise(),
                IAAEmboss(),
                IAASharpen(),
                MotionBlur(),
                RandomBrightnessContrast(),
            ], p=self.p),
            "spatial_level": OneOf([
                HorizontalFlip(),
                Rotate(40)
            ], p=self.p),
        }

    def transform(self):
        """"""
        transforms = self.transforms[self.transform_type]

        if self.label_type == 'kp':
            return Compose(transforms, p=self.p, keypoint_params=KeypointParams(**self.params))
