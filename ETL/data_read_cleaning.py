import pandas as pd

def clean_data(file_path, output_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Handle missing values
    # For example, fill missing 'fare' with median fare
    df['fare'].fillna(df['fare'].median(), inplace=True)
    
    # If there are other numeric columns with missing values, fill them with appropriate values
    numeric_cols = ['nsmiles', 'passengers', 'fare_lg', 'fare_low']
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Handle categorical columns (if any)
    # For example, fill missing 'carrier_lg' with mode
    categorical_cols = ['carrier_lg', 'carrier_low']
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Save the cleaned data to a new CSV file
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

# Example usage
input_file = 'US_Airline_Fares_DS.csv'
output_file = 'Cleaned_US_Airline_Fares_DS.csv'
clean_data(input_file, output_file)
