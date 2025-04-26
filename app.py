import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import json
from datetime import datetime

# 🛑 MUST BE FIRST Streamlit command!!
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Language translations
translations = {
    "en": {
        "title": "Breast Cancer Prediction System",
        "welcome": "Welcome to the Breast Cancer Prediction System",
        "welcome_desc": "This AI-powered system helps in early detection of breast cancer using machine learning.",
        "features": "Key Features",
        "ai_powered": "AI-Powered",
        "ai_desc": "Advanced machine learning algorithms for accurate predictions",
        "realtime": "Real-time Analysis",
        "realtime_desc": "Instant results with detailed probability scores",
        "user_friendly": "User-Friendly",
        "user_desc": "Intuitive interface for easy data input and results interpretation",
        "make_prediction": "Make a Prediction",
        "enter_data": "Enter Patient Data",
        "predict": "Predict",
        "results": "Prediction Results",
        "diagnosis": "Diagnosis",
        "benign": "Benign",
        "malignant": "Malignant",
        "benign_prob": "Benign Probability",
        "malignant_prob": "Malignant Probability",
        "model_info": "Model Information",
        "model_arch": "Model Architecture",
        "model_desc": "This model uses a Random Forest Classifier optimized through GridSearchCV for maximum accuracy.",
        "feature_importance": "Feature Importance",
        "about": "About",
        "about_title": "About This Project",
        "about_desc": "This Breast Cancer Prediction System is designed to assist healthcare professionals in early detection of breast cancer using machine learning algorithms.",
        "tech_stack": "Technology Stack",
        "disclaimer": "Disclaimer",
        "disclaimer_text": "This application is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.",
        "footer": "© 2024 Breast Cancer Prediction System | Version 1.0",
        "nav_home": "Home",
        "nav_prediction": "Prediction",
        "nav_model": "Model Info",
        "nav_about": "About"
    },
    "es": {
        "title": "Sistema de Predicción de Cáncer de Mama",
        "welcome": "Bienvenido al Sistema de Predicción de Cáncer de Mama",
        "welcome_desc": "Este sistema impulsado por IA ayuda en la detección temprana del cáncer de mama mediante aprendizaje automático.",
        "features": "Características Principales",
        "ai_powered": "Impulsado por IA",
        "ai_desc": "Algoritmos avanzados de aprendizaje automático para predicciones precisas",
        "realtime": "Análisis en Tiempo Real",
        "realtime_desc": "Resultados instantáneos con puntuaciones de probabilidad detalladas",
        "user_friendly": "Fácil de Usar",
        "user_desc": "Interfaz intuitiva para fácil entrada de datos e interpretación de resultados",
        "make_prediction": "Realizar Predicción",
        "enter_data": "Ingrese Datos del Paciente",
        "predict": "Predecir",
        "results": "Resultados de la Predicción",
        "diagnosis": "Diagnóstico",
        "benign": "Benigno",
        "malignant": "Maligno",
        "benign_prob": "Probabilidad de Benigno",
        "malignant_prob": "Probabilidad de Maligno",
        "model_info": "Información del Modelo",
        "model_arch": "Arquitectura del Modelo",
        "model_desc": "Este modelo utiliza un Clasificador de Bosque Aleatorio optimizado mediante GridSearchCV para máxima precisión.",
        "feature_importance": "Importancia de Características",
        "about": "Acerca de",
        "about_title": "Acerca de Este Proyecto",
        "about_desc": "Este Sistema de Predicción de Cáncer de Mama está diseñado para asistir a profesionales de la salud en la detección temprana del cáncer de mama mediante algoritmos de aprendizaje automático.",
        "tech_stack": "Tecnologías Utilizadas",
        "disclaimer": "Descargo de Responsabilidad",
        "disclaimer_text": "Esta aplicación es solo para fines educativos y no debe utilizarse como sustituto del consejo, diagnóstico o tratamiento médico profesional.",
        "footer": "© 2024 Sistema de Predicción de Cáncer de Mama | Versión 1.0",
        "nav_home": "Inicio",
        "nav_prediction": "Predicción",
        "nav_model": "Info del Modelo",
        "nav_about": "Acerca de"
    },
    "ta": {
        "title": "மார்பக புற்றுநோய் கணிப்பு அமைப்பு",
        "welcome": "மார்பக புற்றுநோய் கணிப்பு அமைப்புக்கு வரவேற்கிறோம்",
        "welcome_desc": "இந்த AI-இயக்கப்பட்ட அமைப்பு இயந்திர கற்றல் மூலம் மார்பக புற்றுநோயின் ஆரம்ப கண்டறிதலுக்கு உதவுகிறது.",
        "features": "முக்கிய அம்சங்கள்",
        "ai_powered": "AI-இயக்கப்பட்ட",
        "ai_desc": "துல்லியமான கணிப்புகளுக்கான மேம்பட்ட இயந்திர கற்றல் வழிமுறைகள்",
        "realtime": "நிகழ்நேர பகுப்பாய்வு",
        "realtime_desc": "விரிவான நிகழ்தகவு மதிப்பெண்களுடன் உடனடி முடிவுகள்",
        "user_friendly": "பயனர் நட்பு",
        "user_desc": "தரவு உள்ளீடு மற்றும் முடிவுகளை விளக்குவதற்கான உள்ளுணர்வு இடைமுகம்",
        "make_prediction": "கணிப்பு செய்யவும்",
        "enter_data": "நோயாளி தரவை உள்ளிடவும்",
        "predict": "கணிக்கவும்",
        "results": "கணிப்பு முடிவுகள்",
        "diagnosis": "நோயறிதல்",
        "benign": "நல்லியல்பு",
        "malignant": "தீயியல்பு",
        "benign_prob": "நல்லியல்பு நிகழ்தகவு",
        "malignant_prob": "தீயியல்பு நிகழ்தகவு",
        "model_info": "மாதிரி தகவல்",
        "model_arch": "மாதிரி கட்டமைப்பு",
        "model_desc": "இந்த மாதிரி அதிகபட்ச துல்லியத்திற்காக GridSearchCV மூலம் மேம்படுத்தப்பட்ட ரேண்டம் ஃபாரஸ்ட் கிளாசிஃபையரைப் பயன்படுத்துகிறது.",
        "feature_importance": "அம்ச முக்கியத்துவம்",
        "about": "பற்றி",
        "about_title": "இந்த திட்டம் பற்றி",
        "about_desc": "இந்த மார்பக புற்றுநோய் கணிப்பு அமைப்பு இயந்திர கற்றல் வழிமுறைகள் மூலம் மார்பக புற்றுநோயின் ஆரம்ப கண்டறிதலுக்கு உதவும் வகையில் வடிவமைக்கப்பட்டுள்ளது.",
        "tech_stack": "தொழில்நுட்ப அடுக்கு",
        "disclaimer": "மறுப்பு",
        "disclaimer_text": "இந்த பயன்பாடு கல்வி நோக்கங்களுக்காக மட்டுமே மற்றும் தொழில்முறை மருத்துவ ஆலோசனை, நோயறிதல் அல்லது சிகிச்சைக்கு மாற்றாக பயன்படுத்தப்படக்கூடாது.",
        "footer": "© 2024 மார்பக புற்றுநோய் கணிப்பு அமைப்பு | பதிப்பு 1.0",
        "nav_home": "முகப்பு",
        "nav_prediction": "கணிப்பு",
        "nav_model": "மாதிரி தகவல்",
        "nav_about": "பற்றி"
    }
}

# Custom CSS for styling
st.markdown("""
<style>
    :root {
        --primary-color: #4CAF50;
        --secondary-color: #2196F3;
        --danger-color: #f44336;
        --warning-color: #FFC107;
        --success-color: #4CAF50;
        --text-color: #333333;
        --background-color: #f8f9fa;
    }
    
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .feature-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .prediction-card {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .language-selector {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 1000;
    }
    
    .footer {
        text-align: center;
        padding: 1rem;
        background-color: white;
        border-radius: 10px;
        margin-top: 2rem;
        box-shadow: 0 -2px 4px rgba(0,0,0,0.1);
    }
    
    @media (max-width: 768px) {
        .feature-card, .prediction-card, .metric-card {
            margin-bottom: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Language selection
language = st.sidebar.selectbox("🌐 Language / Idioma / மொழி", ["en", "es", "ta"])
t = translations[language]

# Load data and model
data = load_breast_cancer()
feature_names = data.feature_names

try:
    model = joblib.load('breast_cancer_model.joblib')
    scaler = joblib.load('scaler.joblib')
except:
    st.error("Please run model.py first to train and save the model.")
    st.stop()

# Sidebar navigation with icons
with st.sidebar:
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/logo.svg", width=200)
    selected = option_menu(
        menu_title="Navigation",
        options=[t["nav_home"], t["nav_prediction"], t["nav_model"], t["nav_about"]],
        icons=["house", "activity", "info-circle", "question-circle"],
        default_index=0,
    )

# Home Page
if selected == t["nav_home"]:
    st.title(f"🎗️ {t['title']}")
    st.markdown(f"""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2>{t['welcome']}</h2>
        <p>{t['welcome_desc']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features section
    st.header(t["features"])
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='feature-card'>
            <h3>🤖 {t['ai_powered']}</h3>
            <p>{t['ai_desc']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='feature-card'>
            <h3>📊 {t['realtime']}</h3>
            <p>{t['realtime_desc']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='feature-card'>
            <h3>📱 {t['user_friendly']}</h3>
            <p>{t['user_desc']}</p>
        </div>
        """, unsafe_allow_html=True)

# Prediction Page
elif selected == t["nav_prediction"]:
    st.title(f"🎗️ {t['make_prediction']}")
    
    # Create input form with modern design
    st.markdown(f"""
    <div style='background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h2 style='text-align: center; margin-bottom: 2rem;'>{t['enter_data']}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    # Create input fields for each feature
    input_data = {}
    for i, feature in enumerate(feature_names):
        col = col1 if i < len(feature_names) // 2 else col2
        with col:
            input_data[feature] = st.number_input(
                f"{feature}",
                min_value=0.0,
                max_value=1000.0,
                value=float(np.mean(data.data[:, i])),
                step=0.1,
                help=f"Enter the {feature} value"
            )
    
    # Prediction button with modern design
    if st.button(t["predict"], key="predict_button"):
        # Prepare input data
        input_df = pd.DataFrame([input_data])
        
        # Scale the input data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Display results with modern design
        st.markdown(f"""
        <div style='margin-top: 2rem;'>
            <h2 style='text-align: center;'>{t['results']}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Create three columns for results
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.markdown(f"""
            <div class='prediction-card'>
                <h3>{t['diagnosis']}</h3>
                <h2 style='color: {'#4CAF50' if prediction == 1 else '#f44336'};'>
                    {t['benign'] if prediction == 1 else t['malignant']}
                </h2>
            </div>
            """, unsafe_allow_html=True)
        
        with result_col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{t['benign_prob']}</h3>
                <h2 style='color: #4CAF50;'>{probability[1]:.2%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with result_col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{t['malignant_prob']}</h3>
                <h2 style='color: #f44336;'>{probability[0]:.2%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Create probability visualization
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[1] * 100,
            title={'text': t["benign_prob"]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4CAF50"},
                'steps': [
                    {'range': [0, 50], 'color': "#f44336"},
                    {'range': [50, 100], 'color': "#4CAF50"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "darkblue"}
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Model Info Page
elif selected == t["nav_model"]:
    st.title(f"🎗️ {t['model_info']}")
    
    # Model details
    st.markdown(f"""
    <div style='background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h2>{t['model_arch']}</h2>
        <p>{t['model_desc']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature importance visualization
    st.header(t["feature_importance"])
    try:
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title=t["feature_importance"],
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=600,
            margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'color': "darkblue"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Feature importance visualization is not available for the current model.")

# About Page
else:
    st.title(f"🎗️ {t['about']}")
    
    st.markdown(f"""
    <div style='background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h2>{t['about_title']}</h2>
        <p>{t['about_desc']}</p>
        
        <h3>{t['tech_stack']}</h3>
        <ul>
            <li>Python</li>
            <li>Scikit-learn</li>
            <li>Streamlit</li>
            <li>Plotly</li>
        </ul>
        
        <h3>{t['disclaimer']}</h3>
        <p style='color: #f44336;'>{t['disclaimer_text']}</p>
    </div>
    """, unsafe_allow_html=True)

# Footer with current year
current_year = datetime.now().year
st.markdown("---")
st.markdown(f"""
<div class='footer'>
    <p>© {current_year} {t['title']} | Version 1.0</p>
</div>
""", unsafe_allow_html=True)
