# **Proyecto de Predicción de Tarifas Aéreas**

## **Descripción del Proyecto**
Este proyecto tiene como objetivo desarrollar modelos de aprendizaje automático para predecir las tarifas aéreas en los Estados Unidos. El proyecto incluye dos objetivos principales:

1. **Predicción de Tarifas**: Utilizar un modelo de regresión lineal para predecir los valores exactos de las tarifas de vuelos.
2. **Clasificación de Tarifas**: Implementar un modelo de regresión logística para clasificar si una tarifa superará un umbral específico (por ejemplo, $1000).

Estos modelos están diseñados para ayudar a las aerolíneas a optimizar sus estrategias de precios y proporcionar a los consumidores información sobre las tendencias futuras de las tarifas.

## **Estructura del Proyecto**
El proyecto está organizado en los siguientes archivos principales:

### **1. Archivos de Datos**
- **`US_Airline_Fares_DS.csv`**: Conjunto de datos original que contiene información detallada sobre las tarifas aéreas en varias rutas de los Estados Unidos. Incluye datos sobre el origen y destino del vuelo, aerolíneas, y tarifas.
- **`cleaned_airline_fares.csv`**: Conjunto de datos limpiado y preprocesado, listo para ser utilizado en la modelización. Los valores categóricos han sido mapeados a números y las filas con datos faltantes han sido eliminadas.

### **2. Scripts**

#### **`guide.py`**
Este script contiene diccionarios que son utilizados para mapear datos categóricos, tales como nombres de ciudades, códigos de aeropuertos e identificadores de aerolíneas, a valores numéricos. Este mapeo es necesario para que los modelos de aprendizaje automático puedan procesar correctamente las variables categóricas.

#### **`data_read_cleaning.py`**
Es responsable de la limpieza y preprocesamiento del conjunto de datos original. Las tareas realizadas por este script incluyen:
- Cargar el conjunto de datos original.
- Mapear las variables categóricas a valores numéricos utilizando los diccionarios de `guide.py`.
- Seleccionar las características relevantes para el análisis y la modelización.
- Eliminar filas con valores faltantes.
- Guardar el conjunto de datos preprocesado en `cleaned_airline_fares.csv` para su uso posterior.

#### **`linear_regression_model.py`**
Este script implementa un modelo de regresión lineal para predecir los valores exactos de las tarifas aéreas. Las principales tareas que realiza este script son:
- **Preparación de los datos**: Normalización de características y división en conjuntos de entrenamiento y prueba.
- **Implementación del modelo de regresión lineal**: El modelo se entrena utilizando descenso de gradiente para minimizar el error cuadrático medio (MSE).
- **Evaluación del modelo**: Calcula métricas de rendimiento, como el MSE, y genera visualizaciones para comparar las predicciones del modelo con los valores reales.
- **Visualización**: Incluye gráficas de dispersión que muestran la relación entre las predicciones y los valores observados.

#### **`logistic_regression.py`**
Este script implementa un modelo de regresión logística, que se utiliza para clasificar si una tarifa aérea superará un umbral predefinido (por ejemplo, $1000). Las principales tareas de este script incluyen:
- **Preparación de los datos**: Al igual que en `linear_regression_model.py`, los datos se normalizan y dividen en conjuntos de entrenamiento y prueba.
- **Implementación del modelo de regresión logística**: El modelo utiliza la función sigmoide para predecir la probabilidad de que una tarifa supere el umbral. Se utiliza la pérdida de entropía cruzada como función de costo.
- **Evaluación del modelo**: Calcula métricas como precisión, recall, F1-score, y utiliza matrices de confusión para evaluar el rendimiento del modelo.
- **Visualización**: Genera gráficos de rendimiento que muestran la distribución de las predicciones, así como métricas para evaluar la clasificación.

### **3. Reporte**

- **`Reporte_IA_A01368818.pdf`**: Este archivo contiene un informe detallado del proyecto. El reporte incluye una descripción general del problema, los métodos utilizados para la limpieza y preprocesamiento de los datos, una explicación de los modelos implementados (regresión lineal y logística), y los resultados obtenidos de cada uno de los modelos. También se incluyen gráficos que ilustran el rendimiento de los modelos y una discusión sobre los resultados y posibles mejoras para futuros análisis.

## **Conclusión**
Este proyecto demuestra cómo se pueden aplicar modelos de aprendizaje automático para predecir y clasificar tarifas aéreas en los Estados Unidos. Los scripts proporcionados implementan desde la limpieza de los datos hasta la evaluación de modelos, lo que permite un análisis integral de las tarifas aéreas con el uso de regresión lineal y logística. El reporte adjunto brinda una descripción exhaustiva del proceso y los resultados obtenidos.

