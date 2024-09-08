import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
import tensorflow as tf
import matplotlib.pyplot as plt

# --------------------------
# 1. Load the Cleaned Dataset and Explore Features
# --------------------------

# Load the cleaned dataset
data = pd.read_csv('./dataset/cleaned_airline_fares.csv')

# --------------------------
# 2. Prepare Features and Target for Classification
# --------------------------

# Define a threshold (e.g., median fare) to create binary classes
median_fare = data['fare'].median()
data['fare_class'] = (data['fare'] > median_fare).astype(int)

# Define features (X) and new binary target (y)
X = data.drop(columns=['fare', 'fare_class'])
y = data['fare_class']

# Train-test split (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features (scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------
# 3. Define the Neural Network Model for Classification
# --------------------------

# Build the neural network model
inputs = tf.keras.Input(shape=(X_train_scaled.shape[1],))

# Add layers (you can modify the architecture if necessary)
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dense(16, activation='relu')(x)

# Output layer with 'sigmoid' activation for binary classification
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Create the model
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model with classification-specific parameters
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --------------------------
# 4. Train the Model
# --------------------------

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_scaled, y_test)
)

# --------------------------
# 5. Evaluate the Model and Plot ROC Curve
# --------------------------

# Predict probabilities on the test data
y_pred_prob = model.predict(X_test_scaled).ravel()

# Compute ROC curve and ROC AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='darkorange')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
