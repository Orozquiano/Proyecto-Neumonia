import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K

# Importamos tu módulo anterior para asegurar que la imagen entre igual
from src.data.preprocess_img import preprocess

# NECESARIO: Desactiva la ejecución eager para que K.gradients funcione
tf.compat.v1.disable_eager_execution()

def generate_grad_cam(array, model):
    """
    Genera un mapa de calor (Heatmap) que indica qué partes de la imagen
    fueron determinantes para la clasificación.
    """
    # 1. Preprocesar la imagen
    img_tensor = preprocess(array)
    
    # 2. Predicción inicial
    preds = model.predict(img_tensor)
    argmax = np.argmax(preds[0])
    output = model.output[:, argmax]
    
    # 3. Obtener la última capa convolucional
    # NOTA: "conv10_thisone" es el nombre específico que tus compañeros pusieron en su modelo
    try:
        last_conv_layer = model.get_layer("conv10_thisone")
    except ValueError:
        print("Error: No se encontró la capa 'conv10_thisone'. Verifica el nombre en el modelo.")
        return array # Retorna original si falla

    # 4. Calcular gradientes
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    
    # 5. Función de iteración (Backprop)
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img_tensor])
    
    # 6. Ponderación de filtros
    for i in range(pooled_grads_value.shape[0]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        
    # 7. Generar Heatmap promedio
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1 # Normalizar
    
    # 8. Superponer a la imagen original
    # Redimensionar el heatmap al tamaño de la imagen original (o 512x512)
    heatmap = cv2.resize(heatmap, (array.shape[1], array.shape[0]))
    
    # Convertir a RGB (mapa de calor Jet)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Mezclar (60% original, 40% calor)
    # Aseguramos que 'array' sea uint8 y tenga 3 canales
    if len(array.shape) == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        
    img_bgr = cv2.resize(array, (heatmap.shape[1], heatmap.shape[0]))
    
    superimposed_img = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    
    # Convertir de BGR a RGB para Tkinter
    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)