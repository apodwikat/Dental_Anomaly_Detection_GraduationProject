import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline


class ImageResizer(BaseEstimator, TransformerMixin):
    def __init__(self, width=1935, height=1024):
        self.width = width
        self.height = height

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA) for img in X]


class CLAHEEnhancer(BaseEstimator, TransformerMixin):
    def __init__(self, clip_limit=2.0, tile_grid_size=(16, 16)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        return [clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in X]


def preprocess_image(image):
    pipeline = make_pipeline(
        ImageResizer(width=1935, height=1024),
        CLAHEEnhancer(clip_limit=2.0, tile_grid_size=(16, 16))
    )
    return pipeline.transform([image])[0]