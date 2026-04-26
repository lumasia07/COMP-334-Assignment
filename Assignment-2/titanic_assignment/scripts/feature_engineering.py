import pandas as pd
import numpy as np

def engineer_features(df):
    """Creates derived features, encodes categoricals, and applies transformations."""
    # Family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Title extraction
    if 'Name' in df.columns:
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        df['Title'] = df['Title'].replace(rare_titles, 'Rare')
        df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    
    # Fare per person
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    
    # Encoding
    columns_to_encode = ['Sex', 'Embarked', 'Title']
    existing_cols_to_encode = [col for col in columns_to_encode if col in df.columns]
    df = pd.get_dummies(df, columns=existing_cols_to_encode, drop_first=True)
    
    # Log transform
    df['LogFare'] = np.log1p(df['Fare'])
    
    # Drop unnecessary columns
    cols_to_drop = ['Name', 'Ticket']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    return df

if __name__ == "__main__":
    df = pd.read_csv('../data/train_cleaned.csv')
    df_engineered = engineer_features(df)
    df_engineered.to_csv('../data/train_engineered.csv', index=False)