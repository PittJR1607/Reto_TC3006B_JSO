import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 1. Cargar el conjunto de datos limpiado
# --------------------------
# Se carga el conjunto de datos desde un archivo CSV especificado.
# Contiene columnas relacionadas con tarifas aéreas, aeropuertos y otras variables adicionales.
data = pd.read_csv('./dataset/cleaned_airline_fares.csv')

# --------------------------
# 2. Preparar las características (features) y el objetivo (target)
# --------------------------
# Se definen las características (X) eliminando la columna 'fare', que es la variable objetivo (y) que queremos predecir.
# También se conservan los datos de los aeropuertos (airport_1 y airport_2) para graficar y analizarlos más tarde.
X = data.drop(columns=['fare'])  # 'fare' es la columna objetivo que queremos predecir
y = data['fare']
airport_1 = data['airport_1']
airport_2 = data['airport_2']

# Dividimos el conjunto de datos en conjuntos de entrenamiento (80%) y prueba (20%).
X_train, X_test, y_train, y_test, airport_1_train, airport_1_test = train_test_split(
    X, y, airport_1, test_size=0.2, random_state=42
)

# Normalizamos las características utilizando StandardScaler, lo que asegura que cada característica tenga una media de 0 y una desviación estándar de 1.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------
# 3. Definir el modelo de red neuronal
# --------------------------
# El modelo de red neuronal se define con tres capas densas de tamaños 64, 32 y 16, utilizando funciones de activación ReLU.
# La capa de salida contiene una unidad para la regresión (predicción de tarifas) sin ninguna función de activación.
class MSECallback(tf.keras.callbacks.Callback):
    # Callback personalizado para calcular y mostrar el Error Cuadrático Medio (MSE) en el conjunto de validación al final de cada época.
    # Ayuda a monitorear qué tan bien generaliza el modelo a datos no vistos durante el entrenamiento.
    def __init__(self, validation_data):
        super(MSECallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        print(f'Epoch {epoch+1} \n Validation MSE: {mse}')

# Construcción del modelo con tres capas ocultas densas (64, 32, 16) y una capa de salida para regresión.
inputs = tf.keras.Input(shape=(X_train_scaled.shape[1],))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dense(16, activation='relu')(x)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compilamos el modelo utilizando el optimizador Adam y la función de pérdida de error cuadrático medio (MSE), que es adecuada para regresión.
model.compile(optimizer='adam', loss='mean_squared_error')

# --------------------------
# 4. Entrenar el modelo
# --------------------------
# El modelo se entrena con el conjunto de entrenamiento, y se valida en el conjunto de prueba.
# El callback MSECallback se utiliza para imprimir el MSE al final de cada época.
mse_callback = MSECallback(validation_data=(X_test_scaled, y_test))
history = model.fit(
    X_train_scaled, y_train, 
    epochs=20, 
    batch_size=32, 
    validation_data=(X_test_scaled, y_test), 
    callbacks=[mse_callback]
)

# --------------------------
# 5. Evaluar el modelo
# --------------------------
# Después de entrenar, se hacen predicciones sobre el conjunto de prueba y se calcula el error cuadrático medio final (MSE) y el coeficiente de determinación (R²).
y_pred = model.predict(X_test_scaled)
final_mse = mean_squared_error(y_test, y_pred)
final_r2 = r2_score(y_test, y_pred)

print(f'Final Mean Squared Error (MSE): {final_mse}')
print(f'Final R-squared (R²): {final_r2}')

# --------------------------
# 6. Visualización: Predicciones vs Tarifas Reales
# --------------------------
# Gráfico que compara las tarifas predichas con las tarifas reales. La línea roja discontinua representa una predicción perfecta.
# Este gráfico es útil para inspeccionar visualmente qué tan cerca están las predicciones de los valores reales.
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit')
plt.title('Predicted vs Actual Fares')
plt.xlabel('Actual Fare')
plt.ylabel('Predicted Fare')
plt.legend()
plt.grid(True)
plt.show()

# --------------------------
# 7. Visualización: Pérdida de Entrenamiento y Validación a lo largo de las épocas
# --------------------------
# Este gráfico muestra cómo evoluciona la pérdida (MSE) durante las épocas de entrenamiento.
# Es útil para verificar si el modelo está sobreajustando (por ejemplo, si la pérdida de validación aumenta mientras la pérdida de entrenamiento disminuye).
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# --------------------------
# 8. Visualización: Comparación de Tarifas Reales y Predichas por muestra
# --------------------------
# Un gráfico de dispersión que compara las tarifas reales y predichas para cada muestra en el conjunto de prueba.
# El índice de la muestra se usa en el eje x para proporcionar una visión general rápida de qué tan bien se desempeña el modelo en todas las muestras.
indices = np.arange(len(y_test))
plt.figure(figsize=(10, 6))
plt.scatter(indices, y_test, color='blue', alpha=0.6, label='Actual Fares')
plt.scatter(indices, y_pred, color='red', alpha=0.6, label='Predicted Fares')
plt.title('Comparison of Actual vs Predicted Fares')
plt.xlabel('Sample Index')
plt.ylabel('Fare')
plt.legend()
plt.grid(True)
plt.show()

# --------------------------
# 9. Visualización: Tarifas Reales vs Predichas por Aeropuerto (simple)
# --------------------------
# Este gráfico compara las tarifas reales y predichas según el aeropuerto de origen (`airport_1`).
# Ayuda a identificar cómo se desempeña el modelo en diferentes aeropuertos y rutas.
plt.figure(figsize=(10, 6))
plt.scatter(airport_1_test, y_test, color='blue', alpha=0.6, label='Actual Fares')
plt.scatter(airport_1_test, y_pred, color='red', alpha=0.6, label='Predicted Fares')
plt.title('Actual vs Predicted Fares by Airport (Origin)')
plt.xlabel('Airport 1 (Origin)')
plt.ylabel('Fare')
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)
plt.show()

# --------------------------
# 10. Visualización: Comparación detallada de aeropuertos (con puntos pequeños)
# --------------------------
# Este gráfico utiliza puntos más pequeños para las tarifas reales y predichas por aeropuerto, lo que permite una visualización más detallada.
# Es útil para observar discrepancias más finas entre las predicciones y los valores reales.
plt.figure(figsize=(10, 6))
plt.scatter(airport_1_test, y_test, color='blue', alpha=0.6, label='Actual Fares', s=3)
plt.scatter(airport_1_test, y_pred, color='red', alpha=0.6, label='Predicted Fares', s=3)
plt.title('Actual vs Predicted Fares by Airport (Origin)')
plt.xlabel('Airport 1 (Origin)')
plt.ylabel('Fare')
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------
# MEJORA DEL MODELO CON DROPOUT Y BATCH NORMALIZATION
# ------------------------------------------


# Construir el modelo con Dropout y Batch Normalization
inputs = tf.keras.Input(shape=(X_train_scaled.shape[1],))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(16, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# --------------------------
# 3. Entrenar el modelo
# --------------------------
mse_callback = MSECallback(validation_data=(X_test_scaled, y_test))

history = model.fit(
    X_train_scaled, y_train,
    epochs=20, 
    batch_size=32, 
    validation_data=(X_test_scaled, y_test),
    callbacks=[mse_callback]
)

# --------------------------
# 4. Evaluar el modelo
# --------------------------
y_pred = model.predict(X_test_scaled)
final_mse = mean_squared_error(y_test, y_pred)
final_r2 = r2_score(y_test, y_pred)

print(f'Final Mean Squared Error (MSE): {final_mse}')
print(f'Final R-squared (R²): {final_r2}')

# --------------------------
# 5. Visualización de resultados
# --------------------------

# Predicciones vs Tarifas Reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit')
plt.title('Predicted vs Actual Fares')
plt.xlabel('Actual Fare')
plt.ylabel('Predicted Fare')
plt.legend()
plt.grid(True)
plt.show()

# Pérdida de entrenamiento y validación a lo largo de las épocas
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)
plt.show()