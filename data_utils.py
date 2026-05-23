import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file):
    """
    Loads a dataset from a CSV file.
    """
    return pd.read_csv(file)

def preprocess_data(df, target_col, treatment_col, features):
    """
    Cleans and prepares raw data for uplift modeling.
    
    1. Removes any rows with missing (NaN) values in features, target, or treatment.
    2. Converts any text/categorical columns (like 'gender' or 'region') into numbers
       using LabelEncoder, which XGBoost requires for training.
       
    Returns:
        df_clean: A clean Pandas DataFrame with numeric values.
        encoders: A dictionary of trained LabelEncoder objects so we can map 
                  the numbers back to their original text labels if needed.
    """
    # Create a unique list of all columns we need, preserving their order
    all_cols = list(dict.fromkeys(features + [target_col, treatment_col]))
    
    # Drop rows that have missing values in these columns
    df_clean = df[all_cols].dropna().copy()
    
    encoders = {}
    
    # Run through the features and encode text columns
    for col in features:
        # Check if the column is categorical (object type)
        if df_clean[col].dtype == 'object':
            # Create a label encoder instance
            le = LabelEncoder()
            # Convert text values to numbers (e.g. ['Male', 'Female'] -> [1, 0])
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            # Store the encoder so we can reference class labels later
            encoders[col] = le
            
    return df_clean, encoders
