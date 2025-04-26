import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from lazypredict.Supervised import LazyClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data():
    # Load the breast cancer dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def find_best_model(X_train, y_train, X_test, y_test):
    # Use LazyPredict to compare models
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    print(models)
    
    # Get the best model based on accuracy
    best_model_name = models.index[0]
    return best_model_name

def train_and_optimize_model(X_train, y_train, X_test, y_test, model_name):
    # Import the best model
    from sklearn.ensemble import RandomForestClassifier
    
    # Initialize the model
    model = RandomForestClassifier(random_state=42)
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Perform grid search
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Evaluate the model
    print("Best Parameters:", grid_search.best_params_)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nROC AUC Score:", roc_auc_score(y_test, y_pred_proba))
    
    return best_model

def save_model_and_scaler(model, scaler):
    # Save the model and scaler
    joblib.dump(model, 'breast_cancer_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    
    # Find the best model
    best_model_name = find_best_model(X_train, y_train, X_test, y_test)
    print(f"\nBest model found: {best_model_name}")
    
    # Train and optimize the model
    best_model = train_and_optimize_model(X_train, y_train, X_test, y_test, best_model_name)
    
    # Save the model and scaler
    save_model_and_scaler(best_model, scaler)

if __name__ == "__main__":
    main() 