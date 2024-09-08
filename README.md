# Reto_TC3006C_JSO

Este repositorio contiene un proyecto completo de predicción de tarifas aéreas utilizando técnicas avanzadas de **Machine Learning**, específicamente redes neuronales. Se busca modelar y predecir precios de vuelos en función de diversas características del dataset como la distancia, aerolíneas, y aeropuertos, optimizando el modelo para obtener predicciones precisas y generalizables.

## **Estructura del Repositorio**

### 1. **ETL/guide.py**
- **Descripción**: Este archivo contiene los diccionarios de mapeo que se utilizan para convertir variables categóricas como los códigos de aeropuertos y las aerolíneas en valores numéricos. Estos mapeos son esenciales para el preprocesamiento de los datos antes de entrenar los modelos de machine learning.
- **Contenido**: Diccionarios como `airport1_dict`, `airport2_dict`, `carrier_lg_dict`, `city1_dict`, entre otros, que asignan un número único a cada valor categórico.

### 2. **ETL/data_cleaning.py**
- **Descripción**: Este script es responsable de la limpieza y transformación de los datos crudos del dataset original de tarifas aéreas. Realiza tareas de preprocesamiento como eliminar valores nulos, mapear las categorías con los diccionarios de `guide.py`, y normalizar ciertas variables.
- **Contenido**: Incluye funciones que transforman los datos en un formato adecuado para ser utilizado en el entrenamiento del modelo, generando un archivo limpio llamado `cleaned_airline_fares.csv`.

### 3. **dataset/cleaned_airline_fares.csv**
- **Descripción**: Este archivo CSV contiene los datos ya procesados y listos para ser utilizados en los modelos de machine learning. Ha pasado por un proceso de limpieza y transformación.
- **Contenido**: Contiene variables como la distancia entre aeropuertos, tarifas, número de pasajeros, y aerolíneas, en formato numérico y adecuado para ser ingresado en el modelo.

### 4. **models/neural_network.py**
- **Descripción**: Contiene el código para construir, entrenar y evaluar un modelo de **red neuronal** para predecir tarifas aéreas. Este archivo incluye la arquitectura del modelo, técnicas de regularización como **Dropout** y **Batch Normalization**, y funciones de entrenamiento con **callbacks personalizados**.
- **Contenido**:
  - **Arquitectura**: Modelo con capas densas (`Dense`), funciones de activación **ReLU**, y una capa de salida para regresión.
  - **Entrenamiento**: Se usa el optimizador **Adam** y la función de pérdida de **MSE** (Mean Squared Error).
  - **Evaluación**: Se implementan gráficos de comparación entre las predicciones y los valores reales, así como el gráfico de evolución de la pérdida.

### 5. **models/linear_regression.py**
- **Descripción**: Implementa un modelo de **regresión lineal** para predecir las tarifas aéreas. Este archivo sirve como un modelo base para compararlo con los resultados más complejos de la red neuronal.
- **Contenido**:
  - Modelo de regresión lineal ajustado usando el conjunto de entrenamiento.
  - Evaluación del modelo con métricas como **MSE** y **R²**, con visualización de los resultados.

