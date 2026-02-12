import numpy as np
from image.preprocessor import Preprocessor
from model import gradcam
from model.gradcam import GradCAM
from model.model_loader import ModelLoader

class Predictor:

    LABELS = {
        0: "bacteriana",
        1: "normal",
        2: "viral"
    }

    def __init__(self, model):
        self.model = model
        self.preprocessor = Preprocessor()
        self.gradcam = GradCAM(model)

    def predict(self, array):
        
        # Preprocesar una sola vez
        processed_img = self.preprocessor.preprocess(array)

        # Predicción
        preds = self.model(processed_img, training=False).numpy()
        prediction = np.argmax(preds)
        proba = np.max(preds) * 100
        label = ["bacteriana", "normal", "viral"][prediction]

        # Grad-CAM usando la imagen procesada
        heatmap = self.gradcam.generate(processed_img, original_image=array)

        return label, proba, heatmap

    