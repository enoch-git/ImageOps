# ImageOps

**Video Demo:** [https://youtu.be/biIaqAe3MmA?si=mdqOkfyC3TuVkRqb](https://youtu.be/biIaqAe3MmA?si=mdqOkfyC3TuVkRqb)

## Description
ImageOps is a browser-based image processing tool[cite: 1] developed as a final project for CS50P. It integrates classic image manipulation with modern machine-learning object detection in a single accessible web application[cite: 1].

### Key Features
* **Visual Filters:** The application applies five to six classic visual filters to images, including grayscale, blur, edge detection, inversion, and a sepia matrix[cite: 1].
* **Object Detection:** The tool includes a built-in neural network for object detection[cite: 1].
* **User Interface:** The app uses sliders to allow users to interact with operations easily, such as adjusting blur intensity[cite: 1].

### Technical Details
* **Framework:** The user interface is built and hosted using Streamlit[cite: 1].
* **Function Logic:** Each image operation is implemented as an independent function that takes an input image array and returns an output image array, keeping the core logic separate from the UI[cite: 1].
* **Lightweight AI Model:** The project uses the YOLO version 8 nano model, which only weighs about 6.5 MB[cite: 1]. This ensures that the application can run inference on a standard CPU, allowing users to run the tool on their laptops without needing a GPU[cite: 1].
