import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Page config
st.set_page_config(
    page_title="TOEFL Score Predictor",
    page_icon="📚",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    nn_model = load_model('nn_model_toefl.h5', compile=False)
    le_board = joblib.load('le_board_toefl.pkl')
    le_gender = joblib.load('le_gender_toefl.pkl')
    scaler = joblib.load('scaler_toefl.pkl')
    return nn_model, le_board, le_gender, scaler

nn_model, le_board, le_gender, scaler = load_models()

# Header
st.title("📚 TOEFL Score Predictor")
st.markdown("### Predict your TOEFL score based on your academic performance")
st.markdown("---")

# All Indian Education Boards
boards = [
    'CBSE', 'ICSE', 'CISCE', 'IB', 'NIOS',
    'Maharashtra State Board', 'Tamil Nadu State Board', 'Karnataka State Board',
    'Andhra Pradesh State Board', 'Telangana State Board', 'Kerala State Board',
    'West Bengal State Board', 'Gujarat State Board', 'Rajasthan State Board',
    'Madhya Pradesh State Board', 'Uttar Pradesh State Board', 'Bihar State Board',
    'Odisha State Board', 'Punjab State Board', 'Haryana State Board',
    'Jharkhand State Board', 'Chhattisgarh State Board', 'Assam State Board',
    'Jammu and Kashmir State Board', 'Himachal Pradesh State Board',
    'Uttarakhand State Board', 'Goa State Board', 'Tripura State Board',
    'Meghalaya State Board', 'Manipur State Board', 'Nagaland State Board',
    'Mizoram State Board', 'Arunachal Pradesh State Board', 'Sikkim State Board'
]

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Enter Your Details")
    
    # Input fields
    board = st.selectbox("Board of Education", boards, index=0)
    
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    
    percentage = st.number_input(
        "12th Grade Percentage (%)",
        min_value=35.0,
        max_value=100.0,
        value=75.0,
        step=0.1,
        help="Enter your 12th grade percentage"
    )
    
    english_score = st.number_input(
        "12th English Score (out of 100)",
        min_value=35,
        max_value=100,
        value=80,
        step=1,
        help="Enter your 12th grade English subject score"
    )
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Predict My TOEFL Score", type="primary", use_container_width=True):
        # Prepare input
        board_encoded = le_board.transform([board])[0]
        gender_encoded = le_gender.transform([gender])[0]
        
        # Create feature array
        features = np.array([[
            board_encoded,
            gender_encoded,
            percentage,
            english_score
        ]])
        
        # Scale numerical features
        features_scaled = features.copy()
        features_scaled[:, 2:4] = scaler.transform(features[:, 2:4])
        
        # Get prediction from Neural Network (best model)
        nn_pred = nn_model.predict(features_scaled, verbose=0)[0][0]
        
        # Round to nearest integer (TOEFL scores are whole numbers)
        predicted_score = int(round(nn_pred))
        predicted_score = max(60, min(120, predicted_score))  # Clamp between 60 and 120
        
        # Store in session state
        st.session_state.prediction = {
            'score': predicted_score
        }

with col2:
    st.subheader("🎯 Prediction Results")
    
    if 'prediction' in st.session_state:
        pred = st.session_state.prediction
        
        # Main prediction card
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f59e0b 0%, #dc2626 100%); 
                    padding: 40px; 
                    border-radius: 20px; 
                    text-align: center;
                    color: white;
                    margin-bottom: 20px;'>
            <h1 style='font-size: 80px; margin: 0; font-weight: bold;'>{pred['score']}</h1>
            <h3 style='margin: 10px 0;'>Predicted TOEFL Score</h3>
            <p style='margin: 0; opacity: 0.9;'>out of 120</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Score level
        if pred['score'] >= 110:
            level = "Excellent"
            color = "#9333ea"
            desc = "Outstanding English proficiency"
        elif pred['score'] >= 100:
            level = "Very Good"
            color = "#2563eb"
            desc = "Strong English proficiency"
        elif pred['score'] >= 90:
            level = "Good"
            color = "#16a34a"
            desc = "Good English proficiency"
        elif pred['score'] >= 80:
            level = "Fair"
            color = "#ca8a04"
            desc = "Adequate English proficiency"
        else:
            level = "Limited"
            color = "#ea580c"
            desc = "Basic English proficiency"
        
        st.markdown(f"""
        <div style='background-color: {color}20; 
                    border-left: 5px solid {color}; 
                    padding: 15px; 
                    border-radius: 10px;
                    margin-bottom: 20px;'>
            <h3 style='color: {color}; margin: 0;'>{level}</h3>
            <p style='color: {color}; margin: 5px 0 0 0; opacity: 0.8;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model info
        st.info("🎯 **Predicted using Neural Network Model** - Our best performing model with 76.1% accuracy (R² Score)")
        
        # Score range
        score_min = max(60, pred['score'] - 5)
        score_max = min(120, pred['score'] + 5)
        st.success(f"📊 **Expected Score Range:** {score_min} - {score_max}")
        
        # University requirements info
        st.markdown("### 🎓 Common University Requirements")
        st.markdown("""
        - **Top Universities (MIT, Stanford, Harvard):** 100-110+
        - **Good Universities:** 90-100
        - **Most Universities:** 80-90
        - **Minimum Requirement:** 60-80
        """)
        
    else:
        st.info("👈 Enter your details and click 'Predict' to see your estimated TOEFL score!")
        
        # TOEFL Score levels guide
        st.markdown("### 📚 TOEFL Score Levels")
        
        score_levels = {
            "110-120": ("Excellent", "Top-tier universities, competitive programs"),
            "100-109": ("Very Good", "Most top universities accept this range"),
            "90-99": ("Good", "Many universities accept this range"),
            "80-89": ("Fair", "Minimum for many universities"),
            "60-79": ("Limited", "May need additional English courses"),
        }
        
        for score_range, (level, desc) in score_levels.items():
            st.markdown(f"**{score_range}** - {level}")
            st.caption(desc)
            st.markdown("")

# Footer
st.markdown("---")
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.metric("Model Accuracy", "76.1%", help="R² Score on test data")

with col_info2:
    st.metric("Training Data", "30,000+", help="Student records used for training")

with col_info3:
    st.metric("Boards Supported", "34", help="Indian education boards")

st.markdown("---")
st.caption("Powered by Machine Learning • Neural Network Model")