import numpy as np
import cv2
import tensorflow as tf
from src.data.preprocess_img import preprocess

def generate_grad_cam(array, model):
    """
    Genera mapa de calor (Grad-CAM).
    Versión blindada: Convierte listas a tensores automáticamente.
    """
    # 1. Preprocesar
    img_tensor = preprocess(array)
    
    # 2. Buscar capa convolucional
    try:
        last_conv_layer = model.get_layer("conv10_thisone")
    except ValueError:
        print("⚠️ Error: No se encontró la capa 'conv10_thisone'.")
        return array 

    # 3. Modelo de gradientes
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    # 4. GradientTape
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_tensor, tf.float32)
        outputs_list = grad_model(inputs)
        
        # --- FIX DE INGENIERÍA: EXTRACCIÓN SEGURA ---
        # Keras a veces devuelve [tensor, tensor] o [tensor, [tensor]]
        # Aquí desempacamos con cuidado.
        conv_outputs = outputs_list[0]
        predictions = outputs_list[1]

        # Si 'predictions' resulta ser una lista (el error que te salía), extraemos el tensor
        if isinstance(predictions, list) or isinstance(predictions, tuple):
            predictions = predictions[0]

        # Aseguramos que sea un Tensor de TensorFlow
        predictions = tf.convert_to_tensor(predictions)
        conv_outputs = tf.convert_to_tensor(conv_outputs)
        # --------------------------------------------

        # Ahora sí podemos hacer slicing sin error
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # 5. Calcular Gradientes
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 6. Generar Heatmap
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # 7. Superponer
    heatmap = cv2.resize(heatmap, (array.shape[1], array.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    if len(array.shape) == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        
    img_bgr = cv2.resize(array, (heatmap.shape[1], heatmap.shape[0]))
    superimposed_img = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    
    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)