import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = './dataset/cleaned_airline_fares.csv'
df = pd.read_csv(file_path)

# Select relevant columns for regression
columns_needed = [
    'Year', 'quarter', 'city1', 'city2', 'airport_1', 'airport_2',
    'nsmiles', 'passengers', 'fare', 'carrier_lg', 'fare_lg',
    'carrier_low', 'fare_low'
]
df_cleaned = df[columns_needed]

# Drop any rows with missing values
df_cleaned.dropna(inplace=True)

# Prepare the data for linear regression
X = df_cleaned.drop(['fare'], axis=1).values
y = df_cleaned['fare'].values

# Shuffle the data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Manual train-test split
train_ratio = 0.8
train_size = int(train_ratio * X.shape[0])

X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

# Normalize the features
def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

X_train = normalize(X_train)
X_test = normalize(X_test)

# Linear regression functions
def predict(params, X):
    return np.dot(X, params[:-1]) + params[-1]

def mean_squared_error(params, X, y):
    predictions = predict(params, X)
    errors = (predictions - y) ** 2
    return np.mean(errors)

def gradient_descent(params, X, y, lr, lambda_reg=0.01):
    predictions = predict(params, X)
    errors = predictions - y
    gradient = np.dot(X.T, errors) / len(y)
    params[:-1] -= lr * (gradient + lambda_reg * params[:-1])
    params[-1] -= lr * np.mean(errors)
    return params

# Initialize parameters
params = np.zeros(X_train.shape[1] + 1)

# Learning rate
lr = 0.01

# Number of iterations
epochs = 10000

# Training the model
__errors__ = []
for epoch in range(epochs):
    params = gradient_descent(params, X_train, y_train, lr)
    error = mean_squared_error(params, X_train, y_train)
    __errors__.append(error)
    print(f'Epoch {epoch}, Mean Squared Error: {error}')

# Predictions
preds_train = predict(params, X_train)
preds_test = predict(params, X_test)

# Evaluation of the predictions
mse_train = mean_squared_error(params, X_train, y_train)
mse_test = mean_squared_error(params, X_test, y_test)
print(f'MSE on Train Data: {mse_train}')
print(f'MSE on Test Data: {mse_test}')

# Plot the error over iterations
plt.plot(__errors__)
plt.title('Linear Regression Error per Iteration')
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()

# Scatter plot of predictions vs actual values (Train)
plt.scatter(range(len(preds_train)), preds_train, color='blue', alpha=0.2, label='Predictions (Train)')
plt.scatter(range(len(y_train)), y_train, color='red', alpha=0.2, label='Actual Values (Train)')
plt.title('Predictions vs Actual Values (Train)')
plt.xlabel('Sample Index')
plt.ylabel('Fare Value')
plt.legend()
plt.show()

# Scatter plot of predictions vs actual values (Test)
plt.scatter(range(len(preds_test)), preds_test, color='green', alpha=0.5, label='Predictions (Test)')
plt.scatter(range(len(y_test)), y_test, color='orange', alpha=0.5, label='Actual Values (Test)')
plt.title('Predictions vs Actual Values (Test)')
plt.xlabel('Sample Index')
plt.ylabel('Fare Value')
plt.legend()
plt.show()
