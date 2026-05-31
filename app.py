import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import lime
import lime.lime_image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# Page configuration
st.set_page_config(
    page_title="ADHD Diagnostics Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a hyper-premium, modern dark aesthetic
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    
    <style>
    /* Global Styles */
    .stApp {
        background-color: #09090b;
        background-image: 
            radial-gradient(at 0% 0%, rgba(17, 24, 39, 1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, rgba(30, 27, 75, 0.5) 0, transparent 50%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Typography */
    h1, h2, h3, h4, p, span, div {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-weight: 800;
        font-size: 3.2rem;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0 5px 0;
        letter-spacing: -1px;
    }
    
    .sub-title {
        font-weight: 400;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 30px;
        letter-spacing: 0.5px;
    }

    /* Premium Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 30px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        margin-bottom: 25px;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border 0.3s ease;
    }
    
    .glass-card:hover {
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.3);
    }

    /* Primary Action Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 14px;
        height: 3.2em;
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        color: white;
        font-weight: 600;
        font-size: 1.05rem;
        border: none;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #6366f1 0%, #3b82f6 100%);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
        color: white;
    }
    
    /* Result Container */
    .result-container {
        padding: 40px 20px;
        text-align: center;
        border-radius: 24px;
        position: relative;
        overflow: hidden;
    }
    
    /* Uploader styling */
    [data-testid="stFileUploadDropzone"] {
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.02);
        border: 2px dashed rgba(255, 255, 255, 0.15);
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border: 2px dashed rgba(56, 189, 248, 0.5);
        background: rgba(56, 189, 248, 0.05);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(9, 9, 11, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Animation */
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-slide-up {
        animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    
    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.05) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# App Header
st.markdown('<h1 class="main-title">🧠 ADHD Diagnostics Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Powered by Advanced Deep Learning & Explainable AI (EfficientNetV2)</p>', unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_my_model():
    model_path = "new_joint_best_model.keras"
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error(f"Model file '{model_path}' not found. Please ensure it is in the same directory as this app.")
        return None

model = load_my_model()

# Preprocessing function
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image)
    raw_img = img_array.copy()
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array.astype(np.float32))
    return img_array, raw_img

def VizGradCAM(model, image, actual_label=None, class_names=['ADHD', 'CONTROL'], interpolant=0.6):
    img_tensor = tf.cast(image, tf.float32)
    img_tensor = preprocess_input(img_tensor)
    if len(img_tensor.shape) == 3:
        img_tensor = tf.expand_dims(img_tensor, axis=0)

    def find_last_conv_layer(layer):
        if hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                res = find_last_conv_layer(sub_layer)
                if res: return res
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
        return None

    try:
        base_model = model.layers[0] if hasattr(model.layers[0], 'layers') else model
        top_layers = model.layers[1:] if base_model != model else []
    except Exception:
        base_model = model
        top_layers = []
    
    target_layer = find_last_conv_layer(base_model)
    if not target_layer: 
        return None

    grad_sub_model = tf.keras.models.Model([base_model.input], [target_layer.output, base_model.output])

    with tf.GradientTape() as tape:
        conv_activations, base_features = grad_sub_model(img_tensor)
        x = base_features
        for layer in top_layers: 
            x = layer(x)
        final_predictions = x
        preds = tf.reshape(final_predictions, (tf.shape(final_predictions)[0], -1))
        prediction_idx = tf.argmax(preds[0])
        loss = preds[:, prediction_idx]

    gradients = tape.gradient(loss, conv_activations)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    
    conv_activations = conv_activations[0]
    heatmap = conv_activations @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    orig = image.copy()
    if orig.max() <= 1.01: 
        orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
        orig = np.uint8(255 * orig)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    superimposed = np.uint8(orig * (1-interpolant) + heatmap_color * interpolant)
    ax.imshow(superimposed)
    ax.axis('off')
    
    return fig

def generate_lime_explanation(model, raw_img, class_names):
    explainer = lime.lime_image.LimeImageExplainer()
    
    def predict_fn(images):
        processed = preprocess_input(images.astype(np.float32))
        preds = model.predict(processed, verbose=0)
        if isinstance(preds, list):
            preds = preds[0]
        return np.reshape(preds, (len(images), -1))

    explanation = explainer.explain_instance(
        raw_img, 
        predict_fn, 
        top_labels=2, 
        hide_color=0, 
        num_samples=500
    )
    
    dict_heatmap = dict(explanation.local_exp[explanation.top_labels[0]])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.imshow(raw_img)
    im = ax.imshow(heatmap_norm, cmap='coolwarm', alpha=0.6)
    ax.axis('off')
    
    return fig

# Main Interface Content
uploaded_file = st.file_uploader("Upload EEG / fMRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Handle state clearing when a new file is uploaded
    if 'current_file' not in st.session_state or st.session_state['current_file'] != uploaded_file.name:
        for k in ['classified', 'last_prediction', 'xai_running', 'gradcam_fig', 'lime_fig']:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state['current_file'] = uploaded_file.name

    col1, col2 = st.columns([1, 1.3], gap="large")
    
    with col1:
        st.markdown('<div class="glass-card animate-slide-up">', unsafe_allow_html=True)
        st.markdown('<h3 style="margin-top:0; font-size: 1.3rem; color:#e2e8f0; font-weight: 600;">📷 Input Signal</h3>', unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        # Classification button
        if model and not st.session_state.get('classified', False):
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Analyze Scan"):
                with st.spinner("Processing neural patterns..."):
                    preprocessed_img, raw_img = preprocess_image(image)
                    predictions = model.predict(preprocessed_img)
                    
                    if isinstance(predictions, list):
                        preds_array = predictions[0]
                    else:
                        preds_array = predictions
                    
                    preds_flat = np.squeeze(preds_array)
                    if preds_flat.ndim == 0: 
                        preds_flat = np.array([preds_flat])
                    
                    class_names = ["ADHD", "CONTROL"]
                    predicted_class_idx = int(np.argmax(preds_flat))
                    confidence = float(preds_flat[predicted_class_idx]) * 100
                    
                    st.session_state['last_prediction'] = {
                        'raw_img': raw_img,
                        'class_names': class_names,
                        'label': class_names[predicted_class_idx],
                        'confidence': confidence
                    }
                    st.session_state['classified'] = True
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.get('classified', False):
            lp = st.session_state['last_prediction']
            label = lp['label']
            confidence = lp['confidence']
            
            # Dynamic styling based on prediction
            if label == "ADHD":
                color = "#f43f5e" # Rose-500
                bg_color = "rgba(244, 63, 94, 0.05)"
                border_color = "rgba(244, 63, 94, 0.3)"
                shadow = "0 0 40px rgba(244, 63, 94, 0.15)"
            else:
                color = "#10b981" # Emerald-500
                bg_color = "rgba(16, 185, 129, 0.05)"
                border_color = "rgba(16, 185, 129, 0.3)"
                shadow = "0 0 40px rgba(16, 185, 129, 0.15)"

            # Diagnostic Result Card
            st.markdown(f"""
                <div class="glass-card result-container animate-slide-up" style="
                    background: {bg_color}; 
                    border: 1px solid {border_color}; 
                    box-shadow: {shadow};
                ">
                    <p style="color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; margin: 0;">Diagnostic Inference</p>
                    <h1 style="color: {color}; font-size: 4rem; font-weight: 800; margin: 10px 0; letter-spacing: -2px;">{label}</h1>
                    <div style="background: rgba(0,0,0,0.3); padding: 12px 24px; border-radius: 50px; display: inline-block; border: 1px solid rgba(255,255,255,0.05);">
                        <span style="color: #e2e8f0; font-size: 1.1rem; font-weight: 500;">Confidence Score: <b style="color: {color};">{confidence:.2f}%</b></span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # XAI Dashboard section
            st.markdown('<div class="glass-card animate-slide-up" style="animation-delay: 0.1s;">', unsafe_allow_html=True)
            st.markdown("""
                <h3 style="margin-top:0; font-size: 1.3rem; color:#e2e8f0; font-weight: 600;">🔍 Explainable AI (XAI) Suite</h3>
                <p style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 20px;">Reveal the neural pathways driving the model's decision simultaneously using Grad-CAM and LIME.</p>
            """, unsafe_allow_html=True)
            
            # Check if XAI was already run
            if not st.session_state.get('xai_running', False):
                if st.button("⚡ Run Simultaneous XAI Analysis"):
                    st.session_state['xai_running'] = True
                    st.rerun()
            
            if st.session_state.get('xai_running', False):
                st.markdown("<hr style='opacity:0.1'>", unsafe_allow_html=True)
                
                # We calculate both if they aren't in session state
                if 'gradcam_fig' not in st.session_state:
                    with st.spinner("Computing Grad-CAM Heatmap..."):
                        st.session_state['gradcam_fig'] = VizGradCAM(model, lp['raw_img'], class_names=lp['class_names'])
                
                if 'lime_fig' not in st.session_state:
                    with st.spinner("Computing LIME Attribution..."):
                        try:
                            st.session_state['lime_fig'] = generate_lime_explanation(model, lp['raw_img'], lp['class_names'])
                        except Exception as e:
                            st.error(f"LIME attribution failed: {e}")
                            st.session_state['lime_fig'] = None
                
                # Layout for XAI results
                xai_col1, xai_col2 = st.columns(2)
                
                with xai_col1:
                    st.markdown("""
                        <div style="background: rgba(56, 189, 248, 0.05); padding: 15px; border-radius: 12px; border-left: 4px solid #38bdf8; margin-bottom: 15px;">
                            <h4 style="margin: 0; color: #38bdf8; font-size: 1.1rem;">Grad-CAM</h4>
                            <p style="font-size: 0.85rem; color: #94a3b8; margin: 5px 0 0 0;">Spatial activation heatmap</p>
                        </div>
                    """, unsafe_allow_html=True)
                    if st.session_state.get('gradcam_fig'):
                        st.pyplot(st.session_state['gradcam_fig'], use_container_width=True)
                
                with xai_col2:
                    st.markdown("""
                        <div style="background: rgba(167, 139, 250, 0.05); padding: 15px; border-radius: 12px; border-left: 4px solid #a78bfa; margin-bottom: 15px;">
                            <h4 style="margin: 0; color: #a78bfa; font-size: 1.1rem;">LIME</h4>
                            <p style="font-size: 0.85rem; color: #94a3b8; margin: 5px 0 0 0;">Super-pixel attribution</p>
                        </div>
                    """, unsafe_allow_html=True)
                    if st.session_state.get('lime_fig'):
                        st.pyplot(st.session_state['lime_fig'], use_container_width=True)
                        
            st.markdown('</div>', unsafe_allow_html=True)

# Sidebar Design
with st.sidebar:
    st.markdown("""
        <div style="padding: 10px 0 30px 0;">
            <h2 style="color: #f8fafc; font-weight: 700; font-size: 1.5rem; margin:0;">Diagnostics<span style="color: #38bdf8;">Pro</span></h2>
            <p style="color: #64748b; font-size: 0.9rem; margin-top: 5px;">v3.0.0 Architecture</p>
        </div>
        
        <div style="background: rgba(255, 255, 255, 0.03); padding: 20px; border-radius: 16px; margin-bottom: 25px; border: 1px solid rgba(255, 255, 255, 0.05);">
            <h3 style="color: #e2e8f0; margin-top: 0; font-size: 1.1rem; font-weight: 600;">System Core</h3>
            <p style="font-size: 0.9rem; color: #94a3b8; line-height: 1.6;">Powered by <b>EfficientNetV2</b>, highly optimized for medical spectral analysis and pattern recognition.</p>
        </div>
        
        <div style="background: rgba(255, 255, 255, 0.03); padding: 20px; border-radius: 16px; margin-bottom: 25px; border: 1px solid rgba(255, 255, 255, 0.05);">
            <h3 style="color: #e2e8f0; margin-top: 0; font-size: 1.1rem; font-weight: 600;">Detection Classes</h3>
            <div style="display: flex; align-items: center; margin-bottom: 10px; margin-top: 15px;">
                <div style="width: 12px; height: 12px; border-radius: 50%; background: #f43f5e; margin-right: 10px; box-shadow: 0 0 10px #f43f5e;"></div>
                <span style="color: #cbd5e1; font-size: 0.95rem;">ADHD Positive</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 12px; height: 12px; border-radius: 50%; background: #10b981; margin-right: 10px; box-shadow: 0 0 10px #10b981;"></div>
                <span style="color: #cbd5e1; font-size: 0.95rem;">Control (Negative)</span>
            </div>
        </div>
        
        <div style="background: rgba(244, 63, 94, 0.05); padding: 20px; border-radius: 16px; border: 1px solid rgba(244, 63, 94, 0.2);">
            <h4 style="color: #f43f5e; margin-top: 0; font-size: 1rem; display: flex; align-items: center;">
                <span style="margin-right: 8px;">⚠️</span> Clinical Warning
            </h4>
            <p style="font-size: 0.85rem; color: #fda4af; margin-bottom: 0; line-height: 1.5;">
                Intended for <b>research purposes only</b>. This AI system does not replace certified professional medical diagnosis.
            </p>
        </div>
        
        <div style="margin-top: auto; padding-top: 40px; text-align: center; font-size: 0.8rem; color: #475569;">
            © 2026 AI MedLabs<br>Secure Neural Environment
        </div>
    """, unsafe_allow_html=True)
