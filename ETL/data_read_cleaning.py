
import pandas as pd
from guide import city1_dict, city2_dict, airport1_dict, airport2_dict, carrier_lg_dict, carrier_low_dict

# Load the dataset
file_path = './dataset/US_Airline_Fares_DS.csv'
df = pd.read_csv(file_path)

# Map categorical columns to integers using the dictionaries
df['city1'] = df['city1'].map(city1_dict)
df['city2'] = df['city2'].map(city2_dict)
df['airport_1'] = df['airport_1'].map(airport1_dict)
df['airport_2'] = df['airport_2'].map(airport2_dict)
df['carrier_lg'] = df['carrier_lg'].map(carrier_lg_dict)
df['carrier_low'] = df['carrier_low'].map(carrier_low_dict)

# Select relevant columns for regression
columns_needed = [
    'Year', 'quarter', 'city1', 'city2', 'airport_1', 'airport_2',
    'nsmiles', 'passengers', 'fare', 'carrier_lg', 'fare_lg',
    'carrier_low', 'fare_low'
]
df_cleaned = df[columns_needed]

# Drop any rows with missing values
df_cleaned.dropna(inplace=True)

# Save the cleaned dataset
df_cleaned.to_csv('./dataset/cleaned_airline_fares.csv', index=False)

print('Data cleaning complete! Cleaned data saved to cleaned_airline_fares.csv.')
