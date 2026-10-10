# ImageOps

**Video Demo:** [https://youtu.be/biIaqAe3MmA?si=mdqOkfyC3TuVkRqb](https://youtu.be/biIaqAe3MmA?si=mdqOkfyC3TuVkRqb)

## Project Overview
ImageOps is an interactive, browser-based image processing application designed to democratize both classic computer vision operations and modern deep learning object detection. Built as my final project for CS50P, it eliminates the need to install heavy editing software or write custom scripts just to run a simple neural network or apply visual filters. 

Users can effortlessly upload images and apply up to six distinct visual filters—including grayscale, blur, edge detection, inversion, and a custom sepia matrix[cite: 1]. Beyond basic manipulation, ImageOps features a built-in neural network that performs real-time object detection. The entire experience is wrapped in a clean, intuitive user interface where parameters, like the intensity of a blur kernel, can be adjusted dynamically using native sliders.

## File Structure and Functionality
The repository `enoch-git/imageops` is structured to cleanly separate the application's user interface from its underlying computational logic. 

*   **`project.py`**: This is the core executable file of the application. It houses both the Streamlit user interface routing and the independent image processing functions. Each visual operation is defined here as a discrete Python function.
*   **`test_project.py`**: This file contains the unit tests for the application. Because the image processing logic is decoupled from the UI, this script programmatically tests the core functions (like blurring or grayscale conversions) using `pytest` to ensure mathematical and logical accuracy before deployment.
*   **`yolov8n.pt`**: This is the pre-trained weights file for the YOLO (You Only Look Once) version 8 Nano model. Including this locally allows the application to run object detection without needing to download massive model files at runtime.
*   **`requirements.txt`**: A comprehensive list of the Python dependencies required to run the application, including libraries like Streamlit, OpenCV/Pillow, and Ultralytics.
*   **`.gitignore`**: Ensures that cache files, virtual environments, and system-specific artifacts are not pushed to the repository.
*   **`README.md`**: This documentation file, providing an overview of the project, its files, and the technical decisions behind it.

## Design Choices
Throughout the development of ImageOps, several key design choices were debated and finalized to ensure optimal performance and user experience:

**1. Streamlit for the Frontend Framework**
I debated building a desktop application using Tkinter or PyQt, but ultimately chose Streamlit. Streamlit allowed me to rapidly build a highly interactive, browser-based UI that is inherently cross-platform. It handles file uploading and slider-based interactions cleanly, providing an immediate visual response when a user tweaks an operation.

**2. Decoupling Logic from the UI**
A major architectural decision was implementing each filter and operation as a standalone function that accepts an image array as input and returns a modified image array as output. I actively avoided writing the image processing code directly inside the Streamlit rendering loops. This choice ensures the core logic is highly modular and, most importantly, allows the functions to be imported and tested thoroughly in `test_project.py` without ever launching the app.

**3. Selecting YOLOv8 Nano**
When integrating object detection, I debated using a larger, more accurate model. However, I deliberately chose the YOLOv8 Nano variant (`yolov8n.pt`). Because the Nano model weighs only about 6.5 MB, it performs inference efficiently on a standard CPU. This critical design choice ensures that ImageOps remains lightweight and accessible, allowing anyone to run the tool locally on standard laptops without requiring a high-end GPU.
