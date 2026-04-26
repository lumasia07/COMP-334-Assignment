import pandas as pd
import numpy as np

def clean_data(filepath, save_path=None):
    """Loads raw csv, cleans the data, and optionally saves it."""
    df = pd.read_csv(filepath)
    
    # Impute missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Cabin indicator
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    df = df.drop(columns=['Cabin'])
    
    # Embarked
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Cap outliers in Fare
    fare_99th = df['Fare'].quantile(0.99)
    df['Fare'] = np.where(df['Fare'] > fare_99th, fare_99th, df['Fare'])
    
    # Standardize Sex
    df['Sex'] = df['Sex'].str.lower().str.strip()
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    if save_path:
        df.to_csv(save_path, index=False)
        
    return df

if __name__ == "__main__":
    clean_data('../data/train.csv', '../data/train_cleaned.csv')