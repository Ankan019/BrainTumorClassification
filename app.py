import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dropout, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os
import gdown

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Cerebral AI | Diagnostics", page_icon="🧠", layout="wide")

# --- ADVANCED DARK MODE CSS ---
st.markdown("""
<style>
    /* Hide default Streamlit clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent;}
    
    /* Main App Gradient Background */
    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b, #0f172a);
    }

    /* Style the Sidebar */
    [data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid #334155;
    }
    
    /* Glowing Panels for Images */
    .glow-panel img {
        border-radius: 15px;
        border: 1px solid #3b82f6;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
    }
    
    /* Custom Headers */
    .dashboard-header {
        background: linear-gradient(90deg, #312e81, #1e1b4b);
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #6366f1;
        margin-bottom: 30px;
    }
    .dashboard-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        color: #e0e7ff;
    }
    .dashboard-header p {
        margin: 5px 0 0 0;
        color: #94a3b8;
        font-size: 1.1rem;
    }

    /* File Uploader Restyling */
    .stFileUploader > div > div {
        background-color: rgba(30, 41, 59, 0.5);
        border: 2px dashed #475569;
        border-radius: 15px;
    }
    .stFileUploader > div > div:hover {
        border-color: #6366f1;
        background-color: rgba(49, 46, 129, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: CLINICAL CONTEXT ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3003/3003108.png", width=60) # Placeholder medical icon
    st.title("Cerebral AI Lab")
    st.caption("Dr. Ankan | Diagnostic Terminal")
    st.markdown("---")
    
    st.subheader("📋 Awaiting Scans")
    st.info("🟢 Patient ID: 899-A (Ready)")
    st.warning("🟡 Patient ID: 412-B (Pending)")
    st.warning("🟡 Patient ID: 771-C (Pending)")
    
    st.markdown("---")
    st.subheader("⚙️ Settings")
    st.toggle("High-Res Rendering", value=True)
    st.toggle("Enable Grad-CAM", value=True)

# --- MAIN DASHBOARD HEADER ---
st.markdown("""
<div class="dashboard-header">
    <h1>Brain Tumor detection with AI</h1>
    <p>Upload standard MRI scans or DICOM series for immediate neural network evaluation.</p>
</div>
""", unsafe_allow_html=True)

# --- MODEL LOADING (CACHED) ---
@st.cache_resource
def load_tumor_model():
    
    model_path = 'model.keras'
    
    if not os.path.exists(model_path):
        st.info("Downloading AI model weights (this only happens once)...")
        # PASTE YOUR GOOGLE DRIVE FILE ID HERE:
        file_id = '1PR2axxd0z4SqdjuTjujvNpwDeJMGEaYA' 
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
        
    IMAGE_SIZE = 128
    base_model = VGG16(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights=None)
    model = Sequential([
        Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        base_model,
        Flatten(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(4, activation='softmax')
    ])
    model.load_weights('model.keras')
    return model

model = load_tumor_model()
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- GRAD-CAM EXPLAINABLE AI ---
def generate_gradcam(img_array, full_model, original_image):
    try:
        vgg = full_model.layers[0] 
        last_conv_layer = vgg.get_layer('block5_conv3')
        conv_model = tf.keras.Model(vgg.inputs, last_conv_layer.output)
        
        conv_output_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = vgg.get_layer('block5_pool')(conv_output_input)
        for layer in full_model.layers[1:]:
            x = layer(x)
        top_model = tf.keras.Model(conv_output_input, x)
        
        with tf.GradientTape() as tape:
            conv_outputs = conv_model(img_array)
            tape.watch(conv_outputs)
            preds = top_model(conv_outputs)
            top_class_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_class_index]
            
        grads = tape.gradient(top_class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # Using JET for high contrast in dark mode
        
        original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        superimposed_img = cv2.addWeighted(original_bgr, 0.5, heatmap, 0.5, 0) # Adjusted blending for dark mode
        return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return None

# --- FRONTEND LOGIC ---
uploaded_file = st.file_uploader("Drop MRI Scan Here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Create two primary columns
    col_img, col_data = st.columns([1, 1.2], gap="large")
    
    with col_img:
        st.markdown("### 📷 Source Scan")
        # Added wrapper class for the CSS glowing border
        st.markdown('<div class="glow-panel">', unsafe_allow_html=True)
        st.image(image, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner('Neural Network evaluating...'):
        img_resized = image.resize((128, 128))
        img_array = img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions, axis=1)[0])
        result = class_labels[predicted_class_index]

    with col_data:
        # Using Tabs to keep the interface clean and professional
        tab1, tab2 = st.tabs(["🔬 Diagnostic Output", "🔥 Heatmap Explainability"])
        
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            if confidence < 0.65:
                st.warning("⚠️ **Inconclusive Scan**")
                st.write("Confidence threshold not met. Please verify image quality.")
                st.progress(confidence)
                
            elif result == 'notumor':
                st.success("✅ **Negative for Tumor**")
                st.write("The VGG16 model detected no malignant or benign anomalies.")
                st.progress(confidence)
                st.caption(f"Network Confidence: {confidence * 100:.2f}%")
                
            else:
                st.error(f"🚨 **Positive Detection: {result.capitalize()}**")
                st.write("Anomalous tissue structures detected. See Heatmap tab for locational details.")
                st.progress(confidence)
                st.caption(f"Network Confidence: {confidence * 100:.2f}%")
                
        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            if result == 'notumor' or confidence < 0.65:
                st.info("Heatmap generation is bypassed for negative or inconclusive results.")
            else:
                original_image_np = np.array(image)
                heatmap_img = generate_gradcam(img_array, model, original_image_np)
                
                if heatmap_img is not None:
                    st.markdown('<div class="glow-panel">', unsafe_allow_html=True)
                    st.image(heatmap_img, caption="Grad-CAM Activation Map (Block 5 / Conv 3)", width="stretch")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("Failed to generate Grad-CAM visualization.")