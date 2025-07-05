import streamlit as st
import requests
import numpy as np
from PIL import Image
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from io import BytesIO
import base64

# 🎨 Page Configuration
st.set_page_config(
    page_title="🥔 Potato Health Check",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 🎯 Custom CSS for Beautiful, Simple Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f0f8f0 0%, #e8f5e8 100%);
        min-height: 100vh;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(76, 175, 80, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: white;
        font-size: 1.3rem;
        margin: 1rem 0 0 0;
        opacity: 0.95;
        font-weight: 400;
    }
    
    .upload-container {
        background: white;
        border-radius: 20px;
        padding: 3rem;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border: 3px dashed #4CAF50;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        border-color: #2E7D32;
        box-shadow: 0 12px 35px rgba(0,0,0,0.12);
    }
    
    .upload-icon {
        font-size: 4rem;
        color: #4CAF50;
        margin-bottom: 1rem;
    }
    
    .upload-text {
        font-size: 1.2rem;
        color: #333;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    
    .upload-subtext {
        font-size: 1rem;
        color: #666;
        font-weight: 400;
    }
    
    .result-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    }
    
    .healthy-result {
        border-left: 6px solid #4CAF50;
        background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%);
    }
    
    .disease-result {
        border-left: 6px solid #F44336;
        background: linear-gradient(135deg, #FFEBEE 0%, #FCE4EC 100%);
    }
    
    .result-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .result-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .healthy-title {
        color: #2E7D32;
    }
    
    .disease-title {
        color: #C62828;
    }
    
    .confidence-score {
        display: inline-block;
        background: white;
        padding: 0.8rem 1.5rem;
        border-radius: 50px;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .high-confidence {
        color: #2E7D32;
        border: 2px solid #4CAF50;
    }
    
    .medium-confidence {
        color: #F57C00;
        border: 2px solid #FF9800;
    }
    
    .low-confidence {
        color: #C62828;
        border: 2px solid #F44336;
    }
    
    .info-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #E0E0E0;
    }
    
    .info-card h3 {
        color: #333;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .symptoms-list, .treatment-list {
        list-style: none;
        padding: 0;
    }
    
    .symptoms-list li, .treatment-list li {
        padding: 0.5rem 0;
        border-bottom: 1px solid #F0F0F0;
        color: #555;
        font-weight: 400;
    }
    
    .symptoms-list li:before {
        content: "⚠️ ";
        margin-right: 0.5rem;
    }
    
    .treatment-list li:before {
        content: "💊 ";
        margin-right: 0.5rem;
    }
    
    .healthy-symptoms li:before {
        content: "✅ ";
    }
    
    .healthy-treatment li:before {
        content: "🌱 ";
    }
    
    .severity-indicator {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .severity-high {
        background: #FFEBEE;
        color: #C62828;
        border: 1px solid #F44336;
    }
    
    .severity-medium {
        background: #FFF3E0;
        color: #E65100;
        border: 1px solid #FF9800;
    }
    
    .severity-none {
        background: #E8F5E8;
        color: #2E7D32;
        border: 1px solid #4CAF50;
    }
    
    .action-buttons {
        display: flex;
        gap: 1rem;
        justify-content: center;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .action-btn {
        padding: 0.8rem 2rem;
        border-radius: 50px;
        border: none;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
    }
    
    .btn-primary {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    
    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
    }
    
    .btn-secondary {
        background: white;
        color: #666;
        border: 2px solid #E0E0E0;
    }
    
    .btn-secondary:hover {
        border-color: #4CAF50;
        color: #4CAF50;
    }
    
    .tips-container {
        background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        border-left: 4px solid #4CAF50;
    }
    
    .tips-title {
        color: #2E7D32;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .tip-item {
        background: white;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        color: #333;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 50px;
        border: none;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        font-weight: 600;
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
    }
    
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #E0E0E0;
        font-weight: 400;
    }
    
    /* Mobile Responsive */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.2rem;
        }
        
        .main-header p {
            font-size: 1.1rem;
        }
        
        .upload-container {
            padding: 2rem 1rem;
        }
        
        .result-container {
            padding: 1.5rem;
        }
        
        .action-buttons {
            flex-direction: column;
            align-items: center;
        }
        
        .action-btn {
            width: 100%;
            max-width: 300px;
        }
    }
</style>
""", unsafe_allow_html=True)

# 🥔 Disease Information Database  
DISEASE_INFO = {
    "Early Blight": {
        "name": "Early Blight",
        "description": "A common fungal disease that affects potato leaves, causing brown spots with target-like rings.",
        "symptoms": [
            "Dark brown spots with concentric rings on leaves",
            "Yellowing and wilting of lower leaves first",
            "Spots may have yellow halos around them",
            "Premature leaf drop and reduced yield"
        ],
        "treatment": [
            "Apply fungicide spray (Mancozeb or Chlorothalonil)",
            "Remove and burn infected plant debris",
            "Improve air circulation between plants",
            "Water at soil level, avoid wetting leaves",
            "Use disease-resistant potato varieties"
        ],
        "severity": "Medium",
        "color": "#FF6B35",
        "prevention": "Rotate crops, maintain plant spacing, and monitor humidity levels"
    },
    "Late Blight": {
        "name": "Late Blight",
        "description": "A serious fungal disease that can destroy entire potato crops rapidly, especially in wet conditions.",
        "symptoms": [
            "Water-soaked dark lesions on leaves",
            "White fuzzy growth on leaf undersides",
            "Brown/black patches spreading quickly",
            "Entire plant can die within days",
            "Tubers may develop brown rot"
        ],
        "treatment": [
            "Apply copper-based fungicides immediately",
            "Remove and destroy all infected plants",
            "Improve field drainage systems",
            "Use certified disease-free seed potatoes",
            "Harvest healthy tubers quickly"
        ],
        "severity": "High",
        "color": "#DC3545",
        "prevention": "Plant resistant varieties and monitor weather conditions closely"
    },
    "Healthy": {
        "name": "Healthy Plant",
        "description": "Your potato plant appears healthy with no signs of disease. Continue good farming practices!",
        "symptoms": [
            "Vibrant green leaves with no spots",
            "Strong, upright plant growth",
            "Normal leaf size and color",
            "No wilting or discoloration"
        ],
        "treatment": [
            "Continue regular watering schedule",
            "Monitor plants weekly for changes",
            "Maintain proper fertilization",
            "Keep weeds under control",
            "Ensure adequate sunlight"
        ],
        "severity": "None",
        "color": "#4CAF50",
        "prevention": "Maintain current care routine and stay vigilant"
    }
}

# 🚀 Main App Functions
def get_confidence_class(confidence):
    """Get confidence class for styling"""
    if confidence >= 0.8:
        return "high-confidence"
    elif confidence >= 0.6:
        return "medium-confidence"
    else:
        return "low-confidence"

def get_severity_class(severity):
    """Get severity class for styling"""
    if severity == "High":
        return "severity-high"
    elif severity == "Medium":
        return "severity-medium"
    else:
        return "severity-none"

def predict_disease(image_file):
    """Send image to FastAPI backend for prediction"""
    try:
        files = {"file": ("image.jpg", image_file, "image/jpeg")}
        response = requests.post("http://localhost:8000/predict", files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ Server Error: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("🔴 **Cannot connect to AI server**")
        st.info("Please make sure the API server is running:")
        st.code("python api/main.py", language="bash")
        return None
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        return None

def create_detailed_analysis(all_predictions):
    """Create beautiful detailed analysis using Streamlit components"""
    if not all_predictions:
        return None
    
    # Sort predictions by confidence
    sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
    
    st.markdown("### 🔬 Detailed AI Analysis")
    
    for i, (class_name, confidence) in enumerate(sorted_predictions):
        info = DISEASE_INFO.get(class_name, {})
        disease_name = info.get('name', class_name)
        
        # Determine styling based on confidence and rank
        if i == 0:  # Highest confidence
            icon = "🎯"
            rank_text = "Primary Detection"
            bg_color = "#E8F5E8"
        elif confidence > 0.1:  # Significant confidence
            icon = "⚠️"
            rank_text = "Alternative Possibility"
            bg_color = "#FFF3E0"
        else:  # Low confidence
            icon = "📊"
            rank_text = "Low Probability"
            bg_color = "#F5F5F5"
        
        # Confidence level text
        if confidence >= 0.8:
            conf_text = "Very High"
            conf_color = "#2E7D32"
        elif confidence >= 0.6:
            conf_text = "High"
            conf_color = "#4CAF50"
        elif confidence >= 0.3:
            conf_text = "Medium"
            conf_color = "#FF9800"
        else:
            conf_text = "Low"
            conf_color = "#757575"
        
        # Create card using Streamlit components
        with st.container():
            # Custom CSS for this container
            st.markdown(f"""
            <style>
            .analysis-card-{i} {{
                background: {bg_color};
                border: 2px solid {conf_color};
                border-radius: 15px;
                padding: 1rem;
                margin: 0.5rem 0;
            }}
            </style>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{icon} {disease_name}**")
                st.caption(rank_text)
                st.write(info.get('description', 'No description available'))
            
            with col2:
                st.metric(
                    label="Confidence",
                    value=f"{confidence:.1%}",
                    delta=conf_text
                )
            
            # Progress bar
            st.progress(confidence, text=f"{confidence:.1%} confidence")
            
            st.markdown("---")
    
    return True

def display_result(disease_class, confidence, model_name):
    """Display the main analysis result"""
    info = DISEASE_INFO.get(disease_class, {})
    
    if not info:
        st.error("❌ Disease information not available")
        return
    
    # Determine result type
    is_healthy = disease_class == "Healthy"
    
    # Main result header
    if is_healthy:
        st.success(f"🌱 **{info['name']}** - Confidence: {confidence:.1%}")
    else:
        st.error(f"⚠️ **{info['name']}** - Confidence: {confidence:.1%}")
    
    st.info(f"🤖 **Analyzed by:** {model_name}")
    
    # Disease information in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Description")
        st.write(info['description'])
        
        # Severity
        severity_colors = {"High": "🔴", "Medium": "🟡", "None": "🟢"}
        severity_color = severity_colors.get(info['severity'], "🟡")
        st.markdown(f"**🎯 Severity Level:** {severity_color} {info['severity']} Risk")
    
    with col2:
        # Symptoms
        st.markdown("### 👁️ What to Look For")
        for symptom in info['symptoms']:
            emoji = "✅" if is_healthy else "⚠️"
            st.write(f"{emoji} {symptom}")
        
        # Treatment  
        st.markdown("### 💡 Recommendations")
        for treatment in info['treatment']:
            emoji = "🌱" if is_healthy else "💊"
            st.write(f"{emoji} {treatment}")

def main():
    # 🎯 Beautiful Header
    st.markdown("""
    <div class="main-header">
        <h1>🥔 Potato Health Check</h1>
        <p>AI-Powered Disease Detection for Farmers</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 📸 Upload Section
    uploaded_file = st.file_uploader(
        "",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear photo of a potato leaf"
    )
    
    if uploaded_file is None:
        st.markdown("""
        <div class="upload-container">
            <div class="upload-icon">📸</div>
            <div class="upload-text">Take a Photo of Your Potato Leaf</div>
            <div class="upload-subtext">Upload a clear image for instant AI analysis</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Helpful tips for farmers
        st.markdown("""
        <div class="tips-container">
            <div class="tips-title">💡 Tips for Best Results</div>
            <div class="tip-item">📱 Use good lighting - natural daylight works best</div>
            <div class="tip-item">🎯 Focus on the leaf clearly - avoid blurry photos</div>
            <div class="tip-item">📏 Fill the frame with the leaf for better analysis</div>
            <div class="tip-item">🌿 Include both diseased and healthy parts if possible</div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Display uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = Image.open(uploaded_file)
            st.image(image, caption="Your Potato Leaf", use_column_width=True, 
                    clamp=True)
        
        # Analysis button
        if st.button("🔍 Analyze My Potato Plant", key="analyze"):
            with st.spinner("🤖 Analyzing your plant... Please wait"):
                uploaded_file.seek(0)
                result = predict_disease(uploaded_file)
                
                if result:
                    # Display main result
                    display_result(
                        result['predicted_class'], 
                        result['confidence'],
                        result.get('model_used', 'AI Model')
                    )
                    
                    # Detailed analysis cards
                    if 'all_predictions' in result:
                        create_detailed_analysis(result['all_predictions'])
                    
                    # Action buttons
                    st.markdown('<div class="action-buttons">', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("📸 Check Another Plant", key="another"):
                            st.rerun()
                    
                    with col2:
                        if st.button("📋 Get Full Report", key="report"):
                            st.info("📄 Detailed PDF reports coming soon!")
                    
                    with col3:
                        if st.button("🔄 Refresh", key="refresh"):
                            st.rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Additional farmer guidance
                    disease_class = result['predicted_class']
                    if disease_class != "Healthy":
                        st.markdown("""
                        <div class="tips-container">
                            <div class="tips-title">🚨 Important for Farmers</div>
                            <div class="tip-item">⏰ Act quickly - early treatment is most effective</div>
                            <div class="tip-item">🌾 Check other plants in your field immediately</div>
                            <div class="tip-item">👨‍🌾 Consult your local agricultural extension office</div>
                            <div class="tip-item">📞 Contact a plant pathologist if disease spreads</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("❌ Analysis failed. Please try uploading a different image.")
    
    # Footer
    st.markdown("""
    <div class="footer">
        🥔 Potato Health Check • Helping farmers protect their crops with AI technology<br>
        <small>For educational purposes • Always consult agricultural experts for serious problems</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()