import sys
import os
import numpy as np

from model.predictor_img import Predictor



class DummyLayer:
    def __init__(self, name):
        self.name = name

class DummyModel:
    def __init__(self):
        self.layers = [DummyLayer("conv_1"), DummyLayer("dense_1")]

    def __call__(self, inputs, training=False):
        return np.array([[0.1, 0.8, 0.1]])

def test_predict_returns_valid_output():

    predictor = Predictor(model=DummyModel(), use_gradcam=False)

    fake_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

    label, proba, heatmap = predictor.predict(fake_image)

    assert label in ["bacteriana", "normal", "viral"]
    assert 0 <= proba <= 100
