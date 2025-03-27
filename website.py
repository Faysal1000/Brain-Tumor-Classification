import streamlit as st
import numpy as np
import pickle
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

@tf.keras.utils.register_keras_serializable(package="Custom", name="F1Score")
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * (p * r) / (p + r + tf.keras.backend.epsilon())

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

# Load models
@st.cache_resource
def load_models():
    feature_extractor = tf.keras.models.load_model("feature_extractor.keras")
    
    with open('final_feature_extractor_pipeline.pkl', 'rb') as f:
        feature_selector = pickle.load(f)
    
    with open('mlp_model.pkl', 'rb') as file:
        mlp_loaded = pickle.load(file)
    
    return feature_extractor, feature_selector, mlp_loaded

feature_extractor, feature_selector, mlp_loaded = load_models()

# Image preprocessing
def preprocess_image(image):
    img_resized = cv2.resize(image, IMG_SIZE)
    img = img_resized.astype(np.float32)
    img = (img - np.mean(img)) / (np.std(img) + 1e-8)
    img = (img - np.min(img)) / (np.ptp(img) + 1e-8)  
    img = (img * 255).astype(np.uint8) 
    clahe = cv2.createCLAHE(clipLimit=0.02, tileGridSize=(8, 8))
    img_clahe = np.zeros_like(img)
    for i in range(3):  
        img_clahe[..., i] = clahe.apply(img[..., i])
    img_clahe = img_clahe.astype(np.float32) / 255.0  
    return img_clahe

# Prediction function
def predict_image_class(img, feature_extractor, feature_selector, classifier):
    img_processed = preprocess_image(img)
    features = feature_extractor.predict(tf.expand_dims(img_processed, axis=0), verbose=0)
    selected_features = feature_selector.transform(features)
    prediction_probs = classifier.predict_proba(selected_features)[0]
    return dict(zip(CLASS_NAMES, prediction_probs))

# Streamlit UI
st.title("Brain Tumor Detection from MRI Scans")
st.markdown("""
**Upload an MRI scan for automated tumor classification**  
Supported formats: JPEG, PNG
""")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Read and process image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Make prediction
        predictions = predict_image_class(img_rgb, feature_extractor, feature_selector, mlp_loaded)
        predicted_class = max(predictions, key=predictions.get)
        
        # Create two columns with 3:7 ratio
        col1, col2 = st.columns([3, 7])
        
        with col1:
            st.markdown("<h5 style='margin-bottom:10px;'>MRI Scan Preview</h5>", unsafe_allow_html=True)
            st.image(img_rgb, width=230)  
        
        with col2:
            st.markdown("<h5 style='margin-bottom:10px;'>Classification Confidence</h5>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(9, 4)) 
            values = [predictions[cls] for cls in CLASS_NAMES]
            
            # Horizontal bar plot
            bars = sns.barplot(x=values, y=CLASS_NAMES, palette="mako", ax=ax)
            
            # Plot styling
            ax.set_xlabel("Confidence Score", fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_xticks(np.linspace(0, 1, 5))
            ax.tick_params(axis='both', labelsize=12)
            
            # Value annotations
            for i, value in enumerate(values):
                ax.text(value + 0.02, i, f"{value:.2f}",
                        va='center', ha='left',
                        color='white', fontsize=9,
                        fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
            
            # Clean borders
            ax.spines[['top', 'right', 'bottom']].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)

        # Diagnosis result below
        st.markdown(f"""
        <div style='background-color:#e8f4f8; padding:20px; border-radius:10px; margin-top:20px;'>
            <h3 style='color:#2c3e50; margin-bottom:15px;'> Clinical Diagnosis</h3>
            <p style='font-size:20px; color:#2980b9; margin-bottom:5px;'>
            🧠 <strong>Primary Classification:</strong> {predicted_class}
            </p>
            <p style='font-size:16px; color:#2c3e50;'>
            📈 Confidence Level: {predictions[predicted_class]*100:.1f}%<br>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")