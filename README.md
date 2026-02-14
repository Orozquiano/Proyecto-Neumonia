## Hola! Bienvenido a la herramienta para la detección rápida de neumonía


Deep Learning aplicado en el procesamiento de imágenes radiográficas de tórax en formato DICOM con el fin de clasificarlas en 3 categorías diferentes:

1. Neumonía Bacteriana

2. Neumonía Viral

3. Sin Neumonía

Aplicación de una técnica de explicación llamada Grad-CAM para resaltar con un mapa de calor las regiones relevantes de la imagen de entrada.

---

## Uso de la herramienta:

A continuación le explicaremos cómo empezar a utilizarla.

Requerimientos necesarios para el funcionamiento:

- Instale Anaconda para Windows siguiendo las siguientes instrucciones:
  https://docs.anaconda.com/anaconda/install/windows/

- Abra Anaconda Prompt y ejecute las siguientes instrucciones:

  conda create -n tf tensorflow

  conda activate tf

  cd UAO-Neumonia

  pip install -r requirements.txt

  python detector_neumonia.py

Uso de la Interfaz Gráfica:

- Ingrese la cédula del paciente en la caja de texto
- Presione el botón 'Cargar Imagen', seleccione la imagen del explorador de archivos del computador (Imagenes de prueba en https://drive.google.com/drive/folders/1WOuL0wdVC6aojy8IfssHcqZ4Up14dy0g?usp=drive_link)
- Presione el botón 'Predecir' y espere unos segundos hasta que observe los resultados
- Presione el botón 'Guardar' para almacenar la información del paciente en un archivo excel con extensión .csv
- Presione el botón 'PDF' para descargar un archivo PDF con la información desplegada en la interfaz
- Presión el botón 'Borrar' si desea cargar una nueva imagen

---

## Arquitectura de archivos propuesta.

## detector_neumonia.py

Contiene el diseño de la interfaz gráfica utilizando Tkinter.

Los botones llaman métodos contenidos en otros scripts.

## integrator.py

Es un módulo que integra los demás scripts y retorna solamente lo necesario para ser visualizado en la interfaz gráfica.
Retorna la clase, la probabilidad y una imagen el mapa de calor generado por Grad-CAM.

## read_img.py

Script que lee la imagen en formato DICOM para visualizarla en la interfaz gráfica. Además, la convierte a arreglo para su preprocesamiento.

## preprocess_img.py

Script que recibe el arreglo proveniento de read_img.py, realiza las siguientes modificaciones:

- resize a 512x512
- conversión a escala de grises
- ecualización del histograma con CLAHE
- normalización de la imagen entre 0 y 1
- conversión del arreglo de imagen a formato de batch (tensor)

## load_model.py

Script que lee el archivo binario del modelo de red neuronal convolucional previamente entrenado llamado 'WilhemNet86.h5'.

## grad_cam.py

Script que recibe la imagen y la procesa, carga el modelo, obtiene la predicción y la capa convolucional de interés para obtener las características relevantes de la imagen.

---

## Acerca del Modelo

La red neuronal convolucional implementada (CNN) es basada en el modelo implementado por F. Pasa, V.Golkov, F. Pfeifer, D. Cremers & D. Pfeifer
en su artículo Efcient Deep Network Architectures for Fast Chest X-Ray Tuberculosis Screening and Visualization.

Está compuesta por 5 bloques convolucionales, cada uno contiene 3 convoluciones; dos secuenciales y una conexión 'skip' que evita el desvanecimiento del gradiente a medida que se avanza en profundidad.
Con 16, 32, 48, 64 y 80 filtros de 3x3 para cada bloque respectivamente.

Después de cada bloque convolucional se encuentra una capa de max pooling y después de la última una capa de Average Pooling seguida por tres capas fully-connected (Dense) de 1024, 1024 y 3 neuronas respectivamente.

Para regularizar el modelo utilizamos 3 capas de Dropout al 20%; dos en los bloques 4 y 5 conv y otra después de la 1ra capa Dense.

## Acerca de Grad-CAM

Es una técnica utilizada para resaltar las regiones de una imagen que son importantes para la clasificación. Un mapeo de activaciones de clase para una categoría en particular indica las regiones de imagen relevantes utilizadas por la CNN para identificar esa categoría.

Grad-CAM realiza el cálculo del gradiente de la salida correspondiente a la clase a visualizar con respecto a las neuronas de una cierta capa de la CNN. Esto permite tener información de la importancia de cada neurona en el proceso de decisión de esa clase en particular. Una vez obtenidos estos pesos, se realiza una combinación lineal entre el mapa de activaciones de la capa y los pesos, de esta manera, se captura la importancia del mapa de activaciones para la clase en particular y se ve reflejado en la imagen de entrada como un mapa de calor con intensidades más altas en aquellas regiones relevantes para la red con las que clasificó la imagen en cierta categoría.

## Proyecto original realizado por:

Isabella Torres Revelo - https://github.com/isa-tr
Nicolas Diaz Salazar - https://github.com/nicolasdiazsalazar

---
## Proyecto modificado por:

Luis David Hurtado Caicedo
Manuel Castillo Rosales
Luisa Fernanda Ospina
Juan Pablo Orozco

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





