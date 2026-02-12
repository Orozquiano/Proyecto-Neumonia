import cv2
import numpy as np

class Preprocessor:
    
    def preprocess(self,array):
        self.array = cv2.resize(array, (512, 512))
        self.array = cv2.cvtColor(self.array, cv2.COLOR_BGR2GRAY)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self.array = self.clahe.apply(self.array)
        self.array = self.array / 255
        self.array = np.expand_dims(self.array, axis=-1)
        self.array = np.expand_dims(self.array, axis=0)
        return self.array
