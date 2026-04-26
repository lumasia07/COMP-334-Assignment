import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def select_features(df, target_col='Survived'):
    """Runs correlation analysis and Random Forest feature importance."""
    # We drop ID if present, and target col
    X = df.drop(columns=[target_col, 'PassengerId'], errors='ignore')
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in dataframe.")
        
    y = df[target_col]
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return importance_df

if __name__ == "__main__":
    df = pd.read_csv('../data/train_engineered.csv')
    importances = select_features(df)
    print("Feature Importances:\n", importances)