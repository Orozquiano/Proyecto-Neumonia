from image.preprocess_img import Preprocessor
from model.grad_cam import GradCAM


class Predictor:
    
    def __init__(self, model, use_gradcam=True):
        self.model = model
        self.preprocessor = Preprocessor()
        self.use_gradcam = use_gradcam

        if use_gradcam:
            self.gradcam = GradCAM(model)
