import pydicom
import numpy as np
from PIL import Image
import cv2

def read_dicom_file(path):
    """
    Lee una imagen en formato DICOM y la procesa para visualización y análisis.
    Retorna:
        - img_RGB: Imagen en color para la interfaz gráfica.
        - img2show: Objeto de imagen PIL para visualización.
    """
    # Usamos dcmread que es el estándar actual (read_file está obsoleto)
    try:
        img = pydicom.dcmread(path)
    except AttributeError:
        # Respaldo por si usan una versión muy vieja de pydicom
        img = pydicom.read_file(path)

    img_array = img.pixel_array
    img2show = Image.fromarray(img_array)
    
    # Normalización de la imagen (0 a 255)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    
    # Convertir a RGB para que Tkinter la pueda mostrar bien
    img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    
    return img_RGB, img2show

def read_jpg_file(path):
    """
    Lee una imagen en formato estándar (JPG/PNG) para pruebas rápidas.
    """
    img = cv2.imread(path)
    img_array = np.asarray(img)
    img2show = Image.fromarray(img_array)
    
    # Normalización
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    
    return img2, img2show