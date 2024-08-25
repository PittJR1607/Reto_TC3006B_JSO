# **Proyecto de Predicción de Tarifas Aéreas**

## **Descripción del Proyecto**
Este proyecto tiene como objetivo desarrollar modelos de aprendizaje automático para predecir las tarifas aéreas en los Estados Unidos. El proyecto incluye dos objetivos principales:
1. **Predicción de Tarifas**: Utilizar un modelo de regresión lineal para predecir los valores exactos de las tarifas de vuelos.
2. **Clasificación de Tarifas**: Implementar un modelo de regresión logística para clasificar si una tarifa superará un umbral específico (por ejemplo, $1000).

Estos modelos están diseñados para ayudar a las aerolíneas a optimizar sus estrategias de precios y proporcionar a los consumidores información sobre las tendencias futuras de las tarifas.

## **Estructura del Proyecto**
El proyecto está organizado en los siguientes archivos principales:

### **1. Archivos de Datos**
- **`US_Airline_Fares_DS.csv`**: Conjunto de datos original que contiene información detallada sobre las tarifas aéreas en varias rutas de los Estados Unidos.
- **`cleaned_airline_fares.csv`**: Conjunto de datos limpiado y preprocesado, listo para ser utilizado en la modelización.

### **2. Scripts**
- **`guide.py`**: Contiene los diccionarios utilizados para mapear los datos categóricos (como nombres de ciudades, códigos de aeropuertos e identificadores de aerolíneas) en valores numéricos.
- **`data_read_cleaning.py`**: Script que maneja la lectura, limpieza y preprocesamiento del conjunto de datos original. Este script:
  - Carga el conjunto de datos.
  - Mapea las variables categóricas a valores enteros utilizando los diccionarios predefinidos.
  - Selecciona las características relevantes para el análisis.
  - Elimina las filas con datos faltantes.
  - Guarda el conjunto de datos limpio para análisis posteriores.

### **3. Modelos**
- **`linear_regression_model.py`**: Script que implementa un modelo de regresión lineal para predecir los valores exactos de las tarifas aéreas. El script incluye:
  - Preparación de datos (e.g., normalización, división en conjuntos de entrenamiento y prueba).
  - Funciones para la regresión lineal (predicción, cálculo del error cuadrático medio, descenso de gradiente).
  - Entrenamiento y evaluación del modelo.
  - Visualización del rendimiento del modelo (gráficas de error, gráficas de dispersión de predicciones vs. valores reales).

- **`logistic_regression.py`**: Script que implementa un modelo de regresión logística para clasificar si una tarifa superará un umbral específico. El script incluye:
  - Pasos de preparación de datos similares a los utilizados en `linear_regression_model.py`.
  - Funciones para la regresión logística (predicción usando la función sigmoide, cálculo de la pérdida de entropía cruzada, descenso de gradiente).
  - Entrenamiento y evaluación del modelo.
  - Visualización del rendimiento de la clasificación (precisión, recall, F1-score).

