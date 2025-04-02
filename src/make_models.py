from src import config
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import pickle
import sys
sys.path.append(os.path.abspath('..'))  # Adds the parent directory to sys.path
import logging

def load_data():
    """Loads data from the SQLite database."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    query = f"SELECT cleaned_text, sentiment FROM {config.PROCESSED_TABLE}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['cleaned_text'] = df['cleaned_text'].fillna('') 
    return df

def train_models(grid_search=False):
    """
    Trains a specified model (Random Forest or Logistic Regression) 
    with optional GridSearchCV and saves evaluation metrics to the database.

    Args:
        model_type (str): The type of model to train ('random_forest' or 'logistic_regression'). 
                          Defaults to 'random_forest'.
        grid_search (bool): Whether to perform GridSearchCV for hyperparameter tuning. 
                            Defaults to False.
    """
    logging.info(f"Starting models training. Grid search: {grid_search}")
    
    df = load_data()
    logging.info(f"Data loaded successfully. Shape: {df.shape}")

    # Save original indices before vectorization
    df_indices = df.index

    # Feature extraction
    logging.info("Performing TF-IDF vectorization...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['sentiment']
    logging.info(f"Vectorization complete. Features shape: {X.shape}")

    # Saving model vectorizer
    vectorizer_path = os.path.join(config.MODELS_PATH, "vectorizer.pickle")
    os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True) # Be sure that the directory exists
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    logging.info(f"Vectorizer saved to {vectorizer_path}")

    # Train-test split (preserve indices)
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df_indices, test_size=0.2, random_state=42, stratify=y
    )
    logging.info(f"Train/Test split complete. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    logging.info(f"Training 'random forest' model ...")
    if grid_search:
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        logging.info(f"Starting GridSearchCV with param_grid: {param_grid}")
        grid_search_rf = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search_rf.fit(X_train, y_train)

        best_model = grid_search_rf.best_estimator_
        y_pred = best_model.predict(X_test)
    else:
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
    
    # Saving model random_forest
    with open(os.path.join(config.MODELS_PATH, "random_forest.pickle"), "wb") as file:
        pickle.dump(rf, file)

    # Create a DataFrame for the test set with predictions
    test_df = df.loc[test_idx].copy()  # Copy test set rows
    test_df['prediction'] = y_pred  # Add predictions

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    # Connect to the database
    conn = sqlite3.connect(config.DATABASE_PATH)

    # saving predictions
    test_df.to_sql(config.PREDICTIONS_TABLE, conn, if_exists='replace', index=False)
    
    # saving grid search results
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_sql(config.EVALUATION_TABLE, conn,
                      if_exists='replace', index=False)
    # Commit and close the connection
    conn.commit()
    conn.close()
    logging.info("DB Connection closed.")

    logging.info(f"Training 'logistic regression' model ...")
    if grid_search:
        lr = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'C': [0.1, 1, 10, 50],
            'solver': ['liblinear', 'saga'],
            'penalty': ['l1', 'l2']
        }
        logging.info(f"Starting GridSearchCV with param_grid: {param_grid}")

        grid_search_lr = GridSearchCV(lr, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search_lr.fit(X_train, y_train)

        best_model = grid_search_lr.best_estimator_
        y_pred = best_model.predict(X_test)
    else:
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)

    # Saving model logistic_regression
    with open(os.path.join(config.MODELS_PATH, "logistic_regression.pickle"), "wb") as file:
        pickle.dump(lr, file)

    # Create a DataFrame for the test set with predictions
    test_df = df.loc[test_idx].copy()  # Copy test set rows
    test_df['prediction'] = y_pred  # Add predictions

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    '''
    # Connect to the database
    conn = sqlite3.connect(config.DATABASE_PATH)

    # saving predictions
    test_df.to_sql(config.PREDICTIONS_TABLE, conn, if_exists='replace', index=False)
    
    # saving grid search results
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_sql(config.EVALUATION_TABLE, conn,
                      if_exists='replace', index=False)
    # Commit and close the connection
    conn.commit()
    '''    

    logging.info("Train models function finished.")
    return metrics # Returns evalued metris
