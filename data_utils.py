# data_utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader
import config

# Global state to store transformers and training column list
GLOBAL_STATE = {
    'le': LabelEncoder(),
    'scaler': MinMaxScaler(),
    'train_cols': None # Used to ensure train/test columns align after one-hot encoding
}

def preprocess_dataframe(df, is_train=True):
    """
    Handles cleaning, categorical encoding, label encoding, and normalization.
    Fit transformers only on training data (is_train=True) to prevent data leakage.
    """
    # 1. Separate features and target
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]

    # 2. Identify and Encode Categorical Features (One-Hot Encoding)
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols)

    # 3. Handle data consistency: align columns after one-hot encoding
    if is_train:
        # Fit: Save training column order and names
        GLOBAL_STATE['train_cols'] = X.columns.tolist()
    else:
        # Transform: Align Test data to the columns found in training data
        train_cols = GLOBAL_STATE['train_cols']
        
        # Add missing columns (features present in train but not test)
        missing_cols = set(train_cols) - set(X.columns)
        for c in missing_cols:
            X[c] = 0 
            
        # Drop extra columns (features present in test but not train)
        extra_cols = set(X.columns) - set(train_cols)
        X = X.drop(columns=list(extra_cols))
        
        X = X[train_cols] # Reorder columns to match training data

    # 4. Label Encoding for the Target
    if is_train:
        y_encoded = GLOBAL_STATE['le'].fit_transform(y)
    else:
        # FIX FOR UNSEEN LABELS: Filter out any labels not seen during training
        known_classes = GLOBAL_STATE['le'].classes_
        y_known_mask = y.apply(lambda label: label in known_classes)
        
        # Filter X and y dataframes
        X = X[y_known_mask]
        y = y[y_known_mask]
        
        if y_known_mask.sum() < len(y_known_mask):
            print(f"Warning: Dropped {len(y_known_mask) - y_known_mask.sum()} test samples with unknown labels.")

        # Transform the filtered labels
        y_encoded = GLOBAL_STATE['le'].transform(y)

    # 5. Min-Max Normalization (Eq. 7)
    if is_train:
        X_scaled = GLOBAL_STATE['scaler'].fit_transform(X)
    else:
        X_scaled = GLOBAL_STATE['scaler'].transform(X)

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.LongTensor(y_encoded)

    return X_tensor, y_tensor


def load_and_preprocess_separate_data(train_filepath, test_filepath):
    """
    Main loading function. Loads and preprocesses separate training and testing data.
    """
    # Load and clean training data
    print(f"Loading and processing Training data from {train_filepath}...")
    df_train = pd.read_csv(train_filepath)
    df_train = df_train.replace([np.inf, -np.inf], np.nan).dropna()
    X_train_tensor, y_train_tensor = preprocess_dataframe(df_train, is_train=True)
    
    # Load and clean testing data
    print(f"Loading and processing Testing data from {test_filepath}...")
    df_test = pd.read_csv(test_filepath)
    df_test = df_test.replace([np.inf, -np.inf], np.nan).dropna()
    X_test_tensor, y_test_tensor = preprocess_dataframe(df_test, is_train=False)

    print(f"Train Input Shape: {X_train_tensor.shape}")
    print(f"Test Input Shape: {X_test_tensor.shape}")
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, GLOBAL_STATE['le']


def create_dataloaders(X_train, y_train, X_test, y_test):
    """
    Creates PyTorch DataLoaders from pre-split Tensors.
    """
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False)

    return train_loader, test_loader