import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class LinearRegressionModel:
    def __init__(self):
        self.params = []
        self.__errors__ = []
    
    def h(self, sample):
        """Hypothesis function to predict the outcome."""
        return sum(p * x for p, x in zip(self.params, sample))

    def fit(self, samples, y, alfa=0.01, epochs=1000):
        """Train the model using Gradient Descent."""
        self.params = [0] * len(samples[0])  # Initialize parameters
        samples = self.scaling(samples)  # Scale the samples

        for _ in range(epochs):
            old_params = list(self.params)
            self.params = self.GD(samples, y, alfa)
            self.show_errors(samples, y)
            
            if old_params == self.params:  # Stop if no improvement
                break

    def predict(self, sample):
        sample = [1] + sample  # Add bias term
        return self.h(sample)
    
    def mean_squared_error(self, y_true, y_pred):
        return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)
    
    def GD(self, samples, y, alfa):
        temp = list(self.params)
        for j in range(len(self.params)):
            acum = 0
            for i in range(len(samples)):
                error = self.h(samples[i]) - y[i]
                acum += error * samples[i][j]
            temp[j] -= alfa * (1/len(samples)) * acum
        return temp
    
    def show_errors(self, samples, y):
        error_acum = 0
        for i in range(len(samples)):
            error = self.h(samples[i]) - y[i]
            error_acum += np.float64(error) ** 2
        mean_error = error_acum / len(samples)
        self.__errors__.append(mean_error)
    
    def scaling(self, samples):
        samples = pd.DataFrame(samples).transpose().values.tolist() 
        for i in range(1, len(samples)):  # Skip bias term
            avg = sum(samples[i]) / len(samples[i])
            max_val = max(samples[i])
            for j in range(len(samples[i])):
                samples[i][j] = (samples[i][j] - avg) / max_val
        return pd.DataFrame(samples).transpose().values.tolist()

    def train_test_split(self, df, test_size=0.2):
        df_shuffled = df.sample(frac=1).reset_index(drop=True)  # Shuffle the data
        split_index = int((1 - test_size) * len(df_shuffled))
        train_data = df_shuffled[:split_index]
        test_data = df_shuffled[split_index:]
        return train_data, test_data

# Load the cleaned dataset
df = pd.read_csv('Cleaned_US_Airline_Fares_DS.csv')

# Initialize the linear regression model
model = LinearRegressionModel()

# Define the input (features) and output (target)
X = df[['nsmiles', 'citymarketid_1', 'citymarketid_2']].values.tolist()
y = df['fare'].tolist()

# Split the data into training and testing sets
train_data, test_data = model.train_test_split(df)

# Separate features and target for training and testing sets
X_train = train_data[['nsmiles', 'citymarketid_1', 'citymarketid_2']].values.tolist()
y_train = train_data['fare'].tolist()
X_test = test_data[['nsmiles', 'citymarketid_1', 'citymarketid_2']].values.tolist()
y_test = test_data['fare'].tolist()

# Train the model
model.fit(X_train, y_train, alfa=0.01, epochs=10000)

# Make predictions on the testing set
y_pred = [model.predict(sample) for sample in X_test]

# Calculate the Mean Squared Error
mse = model.mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plotting the training and testing data
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Test Data')
plt.plot(range(len(y_test)), y_pred, color='red', label='Predicted Test Data')
plt.xlabel('Test Data Points')
plt.ylabel('Fare')
plt.title('Linear Regression on Airline Fares')
plt.legend()
plt.show()

# Example of predicting fares for future flights
future_flights = [[1000, 305, 606], [1500, 405, 506], [2000, 305, 806]]  # Example future samples
future_fare_predictions = [model.predict(flight) for flight in future_flights]
print(f"Predicted fares for future flights: {future_fare_predictions}")
