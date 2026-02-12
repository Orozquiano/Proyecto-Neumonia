import cv2
import numpy as np

def preprocess(array):
    """
    Preprocesa la imagen de entrada para que coincida con lo que espera la Red Neuronal.
    Pasos:
    1. Redimensionar a 512x512 px.
    2. Convertir a escala de grises.
    3. Aplicar CLAHE (Mejora de contraste adaptativa).
    4. Normalizar valores entre 0 y 1.
    5. Expandir dimensiones para crear un tensor (batch).
    """
    # 1. Resize
    array = cv2.resize(array, (512, 512))
    
    # 2. Grises
    if len(array.shape) > 2 and array.shape[2] == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    
    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    
    # 4. Normalización
    array = array / 255.0
    
    # 5. Expandir dimensiones (H, W) -> (1, H, W, 1)
    array = np.expand_dims(array, axis=-1)  # Agrega canal de color (1)
    array = np.expand_dims(array, axis=0)   # Agrega dimensión de batch (1)
    
    return array