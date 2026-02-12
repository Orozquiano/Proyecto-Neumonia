import tensorflow as tf

class ModelLoader:
    _model = None

    @classmethod
    def get_model(cls):
        # Cargar el modelo solo una vez
        if cls._model is None:
            cls._model = tf.keras.models.load_model("model/conv_MLP_84.h5")
        return cls._model
