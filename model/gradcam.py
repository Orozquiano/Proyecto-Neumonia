import tensorflow as tf
import numpy as np
import cv2

from image import preprocessor


class GradCAM:

    def __init__(self, model):
        self.model = model
    
    # Obtenemos la última capa convolucional del modelo para usarla en Grad-CAM
    def get_last_conv_layer(self):
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer
        raise ValueError("El modelo no contiene capas convolucionales")
    

    def generate(self, img_array, original_image):
    
        last_conv_layer = self.get_last_conv_layer()

        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [last_conv_layer.output, self.model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap) + 1e-8

        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (512, 512))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        original = cv2.resize(original_image, (512, 512))
        superimposed_img = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        return superimposed_img[:, :, ::-1]

