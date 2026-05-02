import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import io


# IMAGE PROCESSING FUNCTIONS

def apply_grayscale(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def apply_blur(image_array, intensity):
    # Maps 1-100 slider to an odd-numbered kernel size
    kernel_size = int((intensity / 100) * 50)
    if kernel_size % 2 == 0: 
        kernel_size += 1
    kernel_size = max(3, kernel_size)
    return cv2.GaussianBlur(image_array, (kernel_size, kernel_size), 0)

def apply_edge_detection(image_array, intensity):
    # Higher intensity = lower threshold (more sensitive)
    lower_thresh = int(250 - (intensity * 2))
    upper_thresh = lower_thresh + 100
    edges = cv2.Canny(image_array, lower_thresh, upper_thresh)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def apply_inversion(image_array):
    return cv2.bitwise_not(image_array)

def apply_sepia(image_array):
    sepia_matrix = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia_img = cv2.transform(image_array, sepia_matrix)
    return np.clip(sepia_img, 0, 255).astype(np.uint8)


#AI OBJECT DETECTION 

@st.cache_resource
def load_yolo_model():
    # Caches the YOLOv8 Nano weights in RAM to prevent redundant loading
    return YOLO("yolov8n.pt")

def apply_yolo(image_array, conf_intensity):
    model = load_yolo_model()
    # Converts 1-100 slider to a 0.01-1.00 decimal for the NMS threshold
    confidence_threshold = conf_intensity / 100.0
    
    results = model(image_array, conf=confidence_threshold)
    annotated_image = results[0].plot()
    return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# UTILITY FUNCTIONS

def convert_image_to_bytes(image_array):
    # Converts a NumPy matrix into a downloadable PNG byte stream
    img_pil = Image.fromarray(image_array)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue()

# MAIN STREAMLIT UI

st.set_page_config(page_title="ImageOps", page_icon="📸", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #00FFAA;'>ImageOps: Vision & Detection</h1>
    <p style='text-align: center; color: #a3a8b8;'>Upload an image to apply visual filters or run AI object detection.</p>
    <hr>
""", unsafe_allow_html=True)

st.sidebar.markdown("<h2 style='text-align: center;'>Control Panel 🎛️</h2>", unsafe_allow_html=True)
selected_action = st.sidebar.selectbox(
    "Select Action",
    ("None", "Grayscale", "Blur", "Edge Detection", "Color Inversion", "Sepia Filter", "YOLO Object Detection")
)

uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='text-align: center;'>Original Image</h3>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown(f"<h3 style='text-align: center;'>Result: {selected_action}</h3>", unsafe_allow_html=True)
        processed_image = None 
        
        # --- LOGIC ROUTER ---
        if selected_action == "None":
            st.info("👈 Select an action from the sidebar to process the image.")
            st.image(image, use_container_width=True)
            
        elif selected_action == "Grayscale":
            processed_image = apply_grayscale(image_array)
            st.image(processed_image, use_container_width=True)
            
        elif selected_action == "Blur":
            intensity = st.slider("Adjust Blur Intensity:", min_value=1, max_value=100, value=50)
            processed_image = apply_blur(image_array, intensity)
            st.image(processed_image, use_container_width=True)
            
        elif selected_action == "Edge Detection":
            intensity = st.slider("Adjust Detail Sensitivity:", min_value=1, max_value=100, value=50)
            processed_image = apply_edge_detection(image_array, intensity)
            st.image(processed_image, use_container_width=True)
            
        elif selected_action == "Color Inversion":
            processed_image = apply_inversion(image_array)
            st.image(processed_image, use_container_width=True)
            
        elif selected_action == "Sepia Filter":
            processed_image = apply_sepia(image_array)
            st.image(processed_image, use_container_width=True)
            
        elif selected_action == "YOLO Object Detection":
            intensity = st.slider("Set AI Confidence Threshold:", min_value=1, max_value=100, value=30)
            with st.spinner('Neural Network scanning image...'):
                processed_image = apply_yolo(image_array, intensity)
                st.image(processed_image, use_container_width=True)

        # DOWNLOAD LOGIC FOR PROCESSED IMAGES 
        if processed_image is not None:
            st.markdown("---") 
            img_bytes = convert_image_to_bytes(processed_image)
            
            # Format the filename to include the selected action (e.g., ImageOps_Blur.png)
            safe_filename = selected_action.replace(" ", "_")
            
            st.download_button(
                label="💾 Download Result",
                data=img_bytes,
                file_name=f"ImageOps_{safe_filename}.png",
                mime="image/png",
                use_container_width=True
            )