import cv2
import numpy as np
import pydicom as dicom
from PIL import Image

class ImageReader:

    @staticmethod
    def read_dicom(path):
        img = dicom.dcmread(path)
        img_array = img.pixel_array

        img2show = Image.fromarray(img_array)

        img2 = img_array.astype(float)
        img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
        img2 = np.uint8(img2)

        img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

        return img_RGB, img2show

    @staticmethod
    def read_image(path):
        # Leer imagen usando OpenCV
        img = cv2.imread(path)

        # Convertir arreglo a numpy array 
        img_array = np.asarray(img)

        # Crear imagen para mostrar usando PIL
        img2show = Image.fromarray(img_array)

        # Preprocesar imagen para el modelo: normalizar y convertir a uint8
        img2 = img_array.astype(float)
        img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
        img2 = np.uint8(img2)

        return img2, img2show
