import numpy as np
from image.preprocess_img import Preprocessor
from model import grad_cam
from model.grad_cam import GradCAM
from model.load_model import ModelLoader

class Predictor:

    LABELS = {
        0: "bacteriana",
        1: "normal",
        2: "viral"
    }

    def __init__(self, model, use_gradcam=True):
        self.model = model
        self.preprocessor = Preprocessor()
        self.use_gradcam = use_gradcam

        if use_gradcam:
            self.gradcam = GradCAM(model)

    def predict(self, array):
        
        # Preprocesar una sola vez
        processed_img = self.preprocessor.preprocess(array)

        # Predicción
        preds = self.model(processed_img, training=False)
        prediction = np.argmax(preds)
        proba = np.max(preds) * 100
        label = ["bacteriana", "normal", "viral"][prediction]

        # Grad-CAM usando la imagen procesada
        # heatmap = self.gradcam.grad_cam(processed_img, original_image=array)
        if self.use_gradcam:
            heatmap = self.gradcam.grad_cam(processed_img, original_image=array)
        else:
            heatmap = None
        return label, proba, heatmap

    