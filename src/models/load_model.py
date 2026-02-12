import tensorflow as tf
import os

def load_trained_model(path):
    """
    Carga el modelo .h5 desde la ruta especificada.
    """
    # Deshabilitar eager execution es necesario para el Grad-CAM de este proyecto
    tf.compat.v1.disable_eager_execution()
    
    if os.path.exists(path):
        print(f"--- Cargando modelo desde: {path} ---")
        return tf.keras.models.load_model(path)
    else:
        print(f"ERROR: No se encontró el modelo en {path}")
        return None 