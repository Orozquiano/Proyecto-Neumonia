import sys
import os
# Asegurarse de que la raíz del proyecto esté en el path para que 'src' sea reconocible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
try:
    from src.model.predictor_img import Predictor
except ImportError:
    # Intento alternativo si se ejecuta desde dentro de la carpeta tests o si la estructura varía
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
