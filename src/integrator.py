import numpy as np
# Importamos los módulos que acabamos de crear
from src.data.preprocess_img import preprocess
from src.visualizations.grad_cam import generate_grad_cam

def predict_and_visualize(array, model):
    """
    Orquesta el proceso completo: Preprocesamiento, Predicción y Visualización.
    Retorna: Etiqueta, Probabilidad, Imagen Heatmap.
    """
    # 1. Preprocesar
    batch_array_img = preprocess(array)
    
    # 2. Predecir (OPTIMIZADO: Solo predecimos una vez)
    preds = model.predict(batch_array_img) # Guardamos el resultado crudo
    
    prediction_idx = np.argmax(preds)      # Sacamos el índice de esa variable
    proba = np.max(preds) * 100            # Sacamos la probabilidad de esa misma variable
    
    label = ""
    # Mapeo según el orden de entrenamiento
    if prediction_idx == 0:
        label = "Bacteriana"
    elif prediction_idx == 1:
        label = "Normal"
    elif prediction_idx == 2:
        label = "Viral"
        
    # 3. Generar Grad-CAM
    try:
        heatmap = generate_grad_cam(array, model)
    except Exception as e:
        print(f"⚠️ Advertencia - Error en Grad-CAM: {e}")
        # Si falla, devolver la imagen original para que el programa no se rompa
        # Aseguramos que sea una imagen válida para Tkinter (aunque sea sin heatmap)
        heatmap = array 
        
    return label, proba, heatmap