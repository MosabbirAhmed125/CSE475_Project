from tensorflow import keras
from tensorflow.keras import layers


def get_basic_augmentation():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )


def get_advanced_augmentation():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(factor=0.2, value_range=(0.0, 1.0)),
            layers.RandomContrast(0.2),
            layers.RandomSaturation(factor=(0.8, 1.2), value_range=(0.0, 1.0)),
            layers.RandomHue(factor=0.05, value_range=(0.0, 1.0)),
            layers.GaussianNoise(0.05),
            layers.RandomErasing(factor=0.2, scale=(0.02, 0.15)),
        ]
    )
