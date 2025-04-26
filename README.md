# Breast Cancer Classification System

This project implements a machine learning-based system for breast cancer classification using the Wisconsin Breast Cancer dataset. The system includes a Streamlit web application for interactive predictions and model visualization.

## Features

- Automated model selection using LazyPredict
- Hyperparameter optimization using GridSearchCV
- Interactive web interface using Streamlit
- Real-time predictions with probability scores
- Feature importance visualization
- Model performance metrics

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd breast-cancer-app
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. First, train the model by running:
```bash
python model.py
```
This will:
- Load and preprocess the breast cancer dataset
- Find the best model using LazyPredict
- Train and optimize the model
- Save the trained model and scaler

2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

## Project Structure

- `model.py`: Contains the machine learning model development code
- `app.py`: Streamlit web application for predictions
- `requirements.txt`: Project dependencies
- `breast_cancer_model.joblib`: Saved trained model
- `scaler.joblib`: Saved feature scaler

## Model Development Process

1. Data Loading and Preprocessing
   - Load the breast cancer dataset from scikit-learn
   - Split into training and testing sets
   - Standardize features using StandardScaler

2. Model Selection
   - Use LazyPredict to compare multiple classification models
   - Select the best-performing model based on accuracy

3. Model Optimization
   - Perform hyperparameter tuning using GridSearchCV
   - Evaluate model performance using various metrics
   - Save the optimized model and scaler

## Web Application Features

- Interactive input form for patient data
- Real-time predictions with probability scores
- Visual representation of prediction probabilities
- Feature importance visualization
- Model information and performance metrics

## Disclaimer

This application is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 