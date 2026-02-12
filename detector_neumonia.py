#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sistema de apoyo diagnóstico para detección de neumonía
mediante redes neuronales convolucionales (CNN).

funcionalidades principales:
- carga de imágenes médicas (jpg, png, dicom)
- preprocesamiento estandarizado
- predicción con modelo entrenado
- generación de mapa de activación (grad-cam)
- visualización de probabilidades por clase con barras y porcentaje

nota:
este sistema es únicamente de apoyo y no sustituye
la evaluación médica profesional.
"""


# importaciones necesarias

from tkinter import *
from tkinter import ttk, font, filedialog
from tkinter.messagebox import askokcancel, showinfo, showwarning
from PIL import ImageTk, Image
import csv
import os
import numpy as np
import tensorflow as tf
import pydicom as dicom
import cv2


# configuración global del sistema


model_path = "model/conv_mlp_84.h5"
labels = ["bacteriana", "normal", "viral"]


# carga del modelo


try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print("error cargando el modelo:", e)
    exit()


# preparación para grad-cam


def get_last_conv_layer(model):
    """ obtiene la última capa convolucional del modelo """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("el modelo no contiene capas convolucionales")

last_conv_layer = get_last_conv_layer(model)

# Modelo auxiliar para cálculo de gradientes
grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[last_conv_layer.output, model.output]
)

# función de preprocesamiento

def preprocess(array):
    """
    preprocesa la imagen:
    - resize 512x512
    - escala de grises
    - clahe
    - normalización
    - expandir dimensiones para batch
    """

    array = cv2.resize(array, (512, 512))

    if len(array.shape) == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)

    array = array.astype("float32") / 255.0
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)

    return array


# grad-cam

def grad_cam(array):
    """
    genera un heatmap sobre la imagen original
    indicando las regiones más relevantes
    """
    img = preprocess(array)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (512,512))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img2 = cv2.resize(array, (512, 512))
    superimposed_img = cv2.addWeighted(img2, 0.6, heatmap, 0.4, 0)

    return superimposed_img[:, :, ::-1]


# función de predicción

def predict(array):
    """
    devuelve:
    - etiqueta predicha
    - probabilidad máxima
    - imagen con grad-cam
    - vector completo de probabilidades
    """

    batch_array_img = preprocess(array)

    preds = model(batch_array_img, training=False).numpy()[0]

    prediction = np.argmax(preds)
    proba = preds[prediction] * 100

    heatmap = grad_cam(array)

    return labels[prediction], proba, heatmap, preds


# lectura de archivos

def read_dicom_file(path):
    img = dicom.dcmread(path)
    img_array = img.pixel_array

    img2show = Image.fromarray(img_array)

    img2 = img_array.astype(float)
    img2 = (np.maximum(img2,0)/img2.max())*255.0
    img2 = np.uint8(img2)
    img_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return img_rgb, img2show

def read_jpg_file(path):
    img = cv2.imread(path)
    img_array = np.asarray(img)

    img2show = Image.fromarray(img_array)

    img2 = img_array.astype(float)
    img2 = (np.maximum(img2,0)/img2.max())*255.0
    img2 = np.uint8(img2)

    return img2, img2show


# interfaz gráfica

class App:

    def __init__(self):

        self.root = Tk()
        self.root.title("Sistema de Apoyo Diagnóstico - Neumonía")
        self.root.geometry("900x620")
        self.root.resizable(False, False)

        fonti = font.Font(weight="bold")

        ttk.Label(self.root, text="Imagen Radiográfica", font=fonti).place(x=120, y=65)
        ttk.Label(self.root, text="Imagen con Heatmap", font=fonti).place(x=560, y=65)

        ttk.Label(self.root,
                  text="Este sistema es de apoyo diagnóstico y no sustituye evaluación médica.",
                  foreground="red").place(x=160, y=580)

        
        # barras de probabilidad por clase con labels
        
        ttk.Label(self.root, text="probabilidades por clase:", font=fonti).place(x=560, y=350)

        # bacteriana
        ttk.Label(self.root, text="bacteriana").place(x=560, y=380)
        self.prob_bac = ttk.Progressbar(self.root, length=180, maximum=100)
        self.prob_bac.place(x=660, y=380)
        self.label_bac_value = ttk.Label(self.root, text="0.00%")
        self.label_bac_value.place(x=850, y=380)

        # normal
        ttk.Label(self.root, text="normal").place(x=560, y=410)
        self.prob_norm = ttk.Progressbar(self.root, length=180, maximum=100)
        self.prob_norm.place(x=660, y=410)
        self.label_norm_value = ttk.Label(self.root, text="0.00%")
        self.label_norm_value.place(x=850, y=410)

        # viral
        ttk.Label(self.root, text="viral").place(x=560, y=440)
        self.prob_vir = ttk.Progressbar(self.root, length=180, maximum=100)
        self.prob_vir.place(x=660, y=440)
        self.label_vir_value = ttk.Label(self.root, text="0.00%")
        self.label_vir_value.place(x=850, y=440)

        self.array = None

        ttk.Button(self.root, text="cargar imagen", command=self.load_img_file).place(x=100, y=500)
        ttk.Button(self.root, text="predecir", command=self.run_model).place(x=250, y=500)

        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)

        self.text_img1.place(x=80, y=100)
        self.text_img2.place(x=520, y=100)

        self.root.mainloop()


    def load_img_file(self):
        filepath = filedialog.askopenfilename(
            title="select image",
            filetypes=(("dicom", "*.dcm"), ("jpeg", "*.jpg"), ("png", "*.png"))
        )

        if filepath:
            if filepath.lower().endswith(".dcm"):
                self.array, img2show = read_dicom_file(filepath)
            else:
                self.array, img2show = read_jpg_file(filepath)

            self.img1 = ImageTk.PhotoImage(
                img2show.resize((250,250), Image.Resampling.LANCZOS)
            )

            self.text_img1.delete("1.0", END)
            self.text_img1.image_create(END, image=self.img1)


    def run_model(self):

        if self.array is None:
            showinfo("Error", "Debe cargar una imagen primero.")
            return

        label, proba, heatmap, preds = predict(self.array)

        # actualizar barras
        self.prob_bac["value"] = preds[0] * 100
        self.prob_norm["value"] = preds[1] * 100
        self.prob_vir["value"] = preds[2] * 100

        # actualizar etiquetas numéricas
        self.label_bac_value.config(text=f"{preds[0]*100:.2f}%")
        self.label_norm_value.config(text=f"{preds[1]*100:.2f}%")
        self.label_vir_value.config(text=f"{preds[2]*100:.2f}%")

        img2 = Image.fromarray(heatmap)
        self.img2 = ImageTk.PhotoImage(
            img2.resize((250,250), Image.Resampling.LANCZOS)
        )
        self.text_img2.delete("1.0", END)
        self.text_img2.image_create(END, image=self.img2)

        showinfo("resultado", f"predicción: {label}\nprobabilidad: {proba:.2f}%")




# Correr el codigo


def main():
    App()

if __name__ == "__main__":
    main()
