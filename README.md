## Hola! Bienvenido a la herramienta para la detección rápida de neumonía

Deep Learning aplicado en el procesamiento de imágenes radiográficas de tórax en formato DICOM con el fin de clasificarlas en 3 categorías diferentes:

1. Neumonía Bacteriana

2. Neumonía Viral

3. Sin Neumonía

Aplicación de una técnica de explicación llamada Grad-CAM para resaltar con un mapa de calor las regiones relevantes de la imagen de entrada.

---

## Uso de la herramienta:

A continuación le explicaremos cómo empezar a utilizarla.

Sigue estas instrucciones para configurar el entorno de desarrollo y ejecutar la aplicación en tu máquina local.

Prerrequisitos
Python 3.11.x (Recomendado para asegurar compatibilidad con Tkinter/Tix).

Git (Opcional, para clonar el repositorio).

Paso 1: Descargar el Código Fuente
Abre tu terminal (PowerShell o CMD) y clona el repositorio. Si no usas Git, puedes descargar el archivo .ZIP desde el botón verde "Code" en GitHub.

Bash
git clone https://github.com/Orozquiano/Proyecto-Neumonia.git
cd Proyecto-Neumonia

Paso 2: Configurar el Entorno Virtual (Recomendado)
Para evitar conflictos de versiones con otras librerías en tu PC, crearemos un entorno aislado.

En Windows:

- Crear el entorno
python -m venv venv

- Activar el entorno
.\venv\Scripts\activate
Nota: Verás (venv) al inicio de tu línea de comandos si se activó correctamente.

- Paso 3: Instalar Dependencias
Instala todas las librerías necesarias (TensorFlow, OpenCV, Numpy, etc.) ejecutando:
pip install -r requirements.txt
pip install tkcap
- Paso 4: Carga Manual del Modelo
Debido a las políticas de tamaño de GitHub (máx. 100MB), el archivo binario del modelo entrenado no está incluido en el repositorio.

Localiza el archivo conv_MLP_84.h5 (Entregado vía USB/Drive).

Copia el archivo y Pégalo dentro de la carpeta models/ del proyecto.

Ruta final requerida: .../Proyecto-Neumonia/models/conv_MLP_84.h5

- Paso 5: Ejecutar la Aplicación
Con el entorno activado y el modelo en su lugar, lanza la interfaz gráfica: python main.py
---

## Arquitectura de archivos propuesta.

Arquitectura de Archivos Propuesta
El proyecto ha migrado de una estructura monolítica a una Arquitectura Modular basada en principios de Clean Code y Separation of Concerns (SoC). A continuación, se detalla la responsabilidad de cada módulo:

- src/data/ (Ingesta y Preprocesamiento)
read_img.py: Módulo de abstracción de entrada. Detecta automáticamente el formato del archivo. Si es DICOM, utiliza pydicom para extraer la matriz de píxeles y normalizar la profundidad de bits (de 12/16 bits a 8 bits). Si es imagen estándar, utiliza OpenCV.

- preprocess_img.py: Pipeline de transformación matemática. Estandariza todas las entradas a 512x512 px, convierte a escala de grises y aplica CLAHE (Ecualización de Histograma Adaptativo) para resaltar estructuras óseas y tejido blando antes de la tensorización.

- src/models/ (Gestión de IA)
load_model.py: Encargado del ciclo de vida del modelo. Implementa una carga segura con el parámetro compile=False, lo que permite utilizar modelos entrenados en versiones antiguas de Keras sin generar conflictos de optimizadores en entornos modernos (Python 3.11 / TF 2.x).

- src/visualizations/ (Explicabilidad)
grad_cam.py: Motor de interpretación visual. Implementa el algoritmo Grad-CAM utilizando tf.GradientTape. A diferencia de implementaciones antiguas, este módulo es compatible con Eager Execution, calculando los gradientes de la última capa convolucional para generar mapas de calor precisos sobre la radiografía original.

- src/ (Núcleo)
integrator.py: Actúa como Orquestador (Facade). Centraliza la lógica de negocio, recibiendo la imagen cruda y coordinando las llamadas al preprocesamiento, la inferencia del modelo y la generación del Grad-CAM, devolviendo un resultado estructurado a la interfaz.

(Raíz)
main.py: Capa de presentación (GUI). Desarrollada en Tkinter, maneja exclusivamente la interacción con el usuario y la renderización de resultados, delegando toda la lógica computacional a los módulos de src.

## Acerca del Modelo

El núcleo inteligente del sistema es una Red Neuronal Convolucional (CNN) optimizada para la clasificación de imágenes médicas de tórax.

Archivo: models/conv_MLP_84.h5

Entrada: Tensores de imagen de 512x512x1 (Escala de grises normalizada).

Arquitectura: El modelo utiliza múltiples capas de convolución para extracción de características (bordes, texturas de infiltrados pulmonares) seguidas de capas densas (MLP) para la clasificación final.

Clases de Salida:

Normal: Sin patologías evidentes.

Neumonía Bacteriana: Caracterizada por consolidaciones focales.

Neumonía Viral: Caracterizada por patrones intersticiales difusos.

Estrategia de Inferencia: El modelo se carga en modo "Inferencia Pura" (sin compilador de entrenamiento) para maximizar la compatibilidad y velocidad de respuesta.

## Acerca de Grad-CAM

Para dotar al sistema de Transparencia y Explicabilidad (XAI), se ha integrado la técnica Grad-CAM (Gradient-weighted Class Activation Mapping).

¿Por qué es necesario?
En medicina, no basta con saber qué predice la IA, es crucial saber por qué. Grad-CAM responde a esta pregunta visualizando qué partes de la radiografía "miró" la red neuronal para tomar su decisión.

Implementación Técnica
El módulo src/visualizations/grad_cam.py realiza los siguientes pasos matemáticos en tiempo real:

Rastreo: Utiliza tf.GradientTape para monitorizar la última capa convolucional del modelo.

Ponderación: Calcula los gradientes de la clase predicha (ej. "Viral") respecto a los mapas de características, determinando la importancia de cada neurona.

Mapa de Calor: Genera una superposición visual donde:

Rojo/Amarillo: Alta relevancia (evidencia de patología).

Azul: Baja relevancia (fondo o tejido sano).

Esto permite descartar "falsos positivos" donde la IA podría estar equivocándose al mirar huesos, clavículas o etiquetas externas en lugar de los pulmones.

## Proyecto original realizado por:

Luis David Hurtado Caicedo
Manuel Castillo Rosales
Juan Orozco
Luisa Ospina