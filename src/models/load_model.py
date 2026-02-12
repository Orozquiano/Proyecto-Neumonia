import tensorflow as tf
import os

def load_trained_model(path):
    """
    Carga el modelo permitiendo ejecución moderna.
    """
    if os.path.exists(path):
        print(f"--- Cargando modelo desde: {path} ---")
        try:
            # compile=False es VITAL para evitar conflictos con el entrenamiento viejo
            return tf.keras.models.load_model(path, compile=False)
        except Exception as e:
            print(f"Error cargando el modelo: {e}")
            return None
    else:
        print(f"ERROR: No se encontró el modelo en {path}")
        return None