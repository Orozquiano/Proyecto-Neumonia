import cv2
import numpy as np

class Preprocessor:
    
    def preprocess(self,array):
        # Redimensionar a 512x512, aplicar CLAHE, normalizar y expandir dimensiones
        self.array = cv2.resize(array, (512, 512))

        # Convertir a escala de grises
        self.array = cv2.cvtColor(self.array, cv2.COLOR_BGR2GRAY)

        # Aplicar CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self.array = self.clahe.apply(self.array)

        # Normalizar a [0, 1]
        self.array = self.array / 255

        # Expandir dimensiones para que sea compatible con el modelo (batch_size, height, width, channels)
        self.array = np.expand_dims(self.array, axis=-1)
        self.array = np.expand_dims(self.array, axis=0)
        return self.array
