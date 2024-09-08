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

threshold = 600  #fare threshold

df_cleaned['fare_above_threshold'] = (df_cleaned['fare'] > threshold).astype(int)


X = df_cleaned.drop(['fare', 'fare_above_threshold'], axis=1).values
y = df_cleaned['fare_above_threshold'].values


indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

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

# Logistic regression functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(params, X):
    return sigmoid(np.dot(X, params[:-1]) + params[-1])

def cross_entropy(params, X, y):
    predictions = predict(params, X)
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    errors = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions)
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
epochs = 3000

# Training the model
__errors__ = []
for epoch in range(epochs):
    params = gradient_descent(params, X_train, y_train, lr)
    error = cross_entropy(params, X_train, y_train)
    __errors__.append(error)
    
    print(f'Epoch {epoch}, Error: {error}')

# Predictions
preds_train = predict(params, X_train) >= 0.5
preds_test = predict(params, X_test) >= 0.5

# Evaluation of the predictions
accuracy_train = np.mean(preds_train == y_train)
accuracy_test = np.mean(preds_test == y_test)
print(f'Accuracy on Train Data: {accuracy_train}')
print(f'Accuracy on Test Data: {accuracy_test}')

# Plot the error over iterations
plt.plot(__errors__)
plt.title('Logistic Regression Error per Iteration')
plt.xlabel('Iterations')
plt.ylabel('Cross-Entropy Error')
plt.show()

# Scatter plot for training data
plt.scatter(np.arange(len(preds_train)), preds_train, color='blue', alpha=0.5, label='Predictions (Train)')
plt.scatter(np.arange(len(y_train)), y_train, color='red', alpha=0.5, label='Actual Values (Train)')
plt.title('Scatter Plot of Predictions vs Actual Values (Train)')
plt.xlabel('Sample Index')
plt.ylabel('Binary Outcome')
plt.legend()
plt.show()

# Scatter plot for test data
plt.scatter(np.arange(len(preds_test)) , preds_test, color='green', alpha=0.5, label='Predictions (Test)', marker='o')
plt.scatter(np.arange(len(y_test)) , y_test, color='orange', alpha=0.5, label='Actual Values (Test)', marker='x')
plt.title('Scatter Plot of Predictions vs Actual Values (Test)')
plt.xlabel('Sample Index')
plt.ylabel('Binary Outcome')
plt.legend()
plt.show()
