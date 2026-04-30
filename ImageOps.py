import streamlit as st
from PIL import Image
import cv2
import numpy as np

# ==========================================
# --- HELPER FUNCTIONS FOR IMAGE LOGIC ---
# ==========================================

def apply_grayscale(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def apply_blur(image_array, intensity):
    # 1. Map the 1-100 slider value to a kernel size (roughly up to 50)
    kernel_size = int((intensity / 100) * 50)
    
    # 2. OpenCV requires the kernel to be an ODD number. 
    # If the math gives us an even number, we add 1.
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    # 3. The minimum kernel size must be at least 3x3
    kernel_size = max(3, kernel_size)
    
    return cv2.GaussianBlur(image_array, (kernel_size, kernel_size), 0)

def apply_edge_detection(image_array, intensity):
    # 1. Map the 1-100 slider. 
    # High intensity = highly sensitive (grabs every tiny detail).
    # To make Canny sensitive, we actually need LOWER threshold numbers.
    lower_thresh = int(250 - (intensity * 2)) # At intensity 100, lower = 50
    upper_thresh = lower_thresh + 100         # At intensity 100, upper = 150
    
    edges = cv2.Canny(image_array, lower_thresh, upper_thresh)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def apply_inversion(image_array):
    return cv2.bitwise_not(image_array)

def apply_sepia(image_array):
    sepia_matrix = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia_img = cv2.transform(image_array, sepia_matrix)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    return sepia_img


# ==========================================
# --- MAIN STREAMLIT UI ---
# ==========================================

# 1. Page Configuration MUST be the first Streamlit command
st.set_page_config(page_title="ImageOps", page_icon="📸", layout="wide")

# 2. Custom HTML for a cooler, centered title
st.markdown("""
    <h1 style='text-align: center; color: #00FFAA;'>ImageOps</h1>
    <p style='text-align: center; color: #a3a8b8;'>Upload an image to apply visual filters or run AI object detection.</p>
    <hr>
""", unsafe_allow_html=True)

# 3. The Sidebar Menu
st.sidebar.markdown("<h2 style='text-align: center;'>Control Panel 🎛️</h2>", unsafe_allow_html=True)
st.sidebar.write("Choose your operation below:")

selected_action = st.sidebar.selectbox(
    "Select Action",
    ("None", "Grayscale", "Blur", "Edge Detection", "Color Inversion", "Sepia Filter", "YOLOX Object Detection")
)

# 4. The File Uploader
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

# 5. UI Logic with Side-by-Side Layout
if uploaded_file is not None:
    # Open the image and convert it to a NumPy array for OpenCV
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    
    # Create two columns of equal width for side-by-side comparison
    col1, col2 = st.columns(2)
    
    # Put the original image in the first column
    with col1:
        st.markdown("<h3 style='text-align: center;'>Original Image</h3>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    
    # Put the result in the second column
    with col2:
        st.markdown(f"<h3 style='text-align: center;'>Result: {selected_action}</h3>", unsafe_allow_html=True)
        
        # --- THE LOGIC ROUTER ---
        
        if selected_action == "None":
            st.info("👈 Select an action from the sidebar to process the image.")
            st.image(image, use_container_width=True)
            
        elif selected_action == "Grayscale":
            processed_image = apply_grayscale(image_array)
            st.image(processed_image, use_container_width=True)
            
        elif selected_action == "Blur":
            # Add a slider from 1 to 100, defaulting to 50
            intensity = st.slider("Adjust Blur Intensity:", min_value=1, max_value=100, value=50)
            processed_image = apply_blur(image_array, intensity)
            st.image(processed_image, use_container_width=True)
            
        elif selected_action == "Edge Detection":
            # Add a slider from 1 to 100, defaulting to 50
            intensity = st.slider("Adjust Detail Sensitivity:", min_value=1, max_value=100, value=50)
            processed_image = apply_edge_detection(image_array, intensity)
            st.image(processed_image, use_container_width=True)
            
        elif selected_action == "Color Inversion":
            processed_image = apply_inversion(image_array)
            st.image(processed_image, use_container_width=True)
            
        elif selected_action == "Sepia Filter":
            processed_image = apply_sepia(image_array)
            st.image(processed_image, use_container_width=True)
            
        elif selected_action == "YOLOX Object Detection":
            st.warning("The YOLOX Object Detection logic hasn't been written yet! That's the final boss.")