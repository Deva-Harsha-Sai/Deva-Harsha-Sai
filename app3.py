import streamlit as st
import cv2
import numpy as np
import time
from numba import njit, prange
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Streamlit App Title
st.title('CUDA Python Task Interface')

# Sidebar for Navigation
option = st.sidebar.selectbox(
    'Select a Project',
    ['Homepage', 'Image Processing', 'Financial Computation', 'Real-Time Video Processing']
)

# Homepage Section
if option == 'Homepage':
    st.header('CUDA Python Tasks: Real-Time Video Processing, Financial Computation & Image Processing')
    
    # Basics about CUDA Python
    st.subheader("What is CUDA Python?")
    st.write(""" 
    CUDA Python is a powerful library that allows you to write Python code that runs on GPUs (Graphics Processing Units) using NVIDIA's CUDA platform.
    It provides several libraries like `Numba`, `PyCUDA`, and `CuPy` that help accelerate your programs by offloading computations to the GPU.
    - **Numba**: A JIT compiler that can parallelize Python code and execute it on the GPU.
    - **PyCUDA**: Provides access to NVIDIA's CUDA parallel computation API.
    - **CuPy**: A GPU array library that is similar to NumPy, but with GPU acceleration.
    CUDA Python is used to accelerate a wide range of applications, including machine learning, scientific computing, image processing, and more.
    """)

    # Footer
    st.markdown("""---
    **By**: Deva Harsha Sai Nangunuri
    """)

# Helper Functions for Each Project

# Numba Grayscale function
@njit(parallel=True)
def grayscale_numba(frame):
    gray = np.empty((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    for i in prange(frame.shape[0]):
        for j in prange(frame.shape[1]):
            r, g, b = frame[i, j]
            gray[i, j] = int(0.2989 * r + 0.5870 * g + 0.1140 * b)
    return gray

# Sobel Edge Detection
@njit(parallel=True)
def sobel_edge_detection(gray):
    h, w = gray.shape
    edges = np.zeros((h, w), dtype=np.uint8)

    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    for i in prange(1, h - 1):
        for j in prange(1, w - 1):
            gx = np.sum(Gx * gray[i - 1:i + 2, j - 1:j + 2])
            gy = np.sum(Gy * gray[i - 1:i + 2, j - 1:j + 2])
            edges[i, j] = min(255, int(np.sqrt(gx**2 + gy**2)))

    return edges

# WebRTC Video Processing
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.start_time = time.time()  # To track time when video starts
        self.frames = []  # Store frames for post-processing

    def recv(self, frame):
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        img = frame.to_ndarray(format="bgr24")

        # Capture frames for only 10 seconds
        if elapsed_time <= 10:
            self.frames.append(img)
        else:
            return None  # Stop capturing frames after 10 seconds

        return frame  # Continue to capture until 10 seconds

    def process_video(self):
        # Process the stored frames after capturing
        processed_frames = []
        for img in self.frames:
            # Preprocess Video (e.g., grayscale and edge detection)
            img_resized = cv2.resize(img, (640, 480))  # Resize the video frame
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            
            # Apply Sobel Edge Detection
            edges = sobel_edge_detection(gray)

            # Convert to BGR for display
            img_resized_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            processed_frames.append((img_resized_bgr, edges_bgr))

        return processed_frames

# Video Processing Section
if option == 'Video Processing':
    st.header('Real-Time Video Processing (10 Seconds Recording)')

    # Create video processor for Streamlit WebRTC
    processor = VideoProcessor()
    webrtc_streamer(key="video", video_processor_factory=lambda: processor)

    # Button to start processing video after 10 seconds
    if st.button('Process Video'):
        processed_frames = processor.process_video()
        if processed_frames:
            st.subheader("Processed Video Results")

            # Show processed frames (greyscaled and edge-detected)
            for i, (gray_frame, edge_frame) in enumerate(processed_frames):
                # Display Greyscale Frame
                st.image(gray_frame, caption=f"Greyscale Frame {i+1}", channels="BGR", use_column_width=True)

                # Display Edge Detection Frame
                st.image(edge_frame, caption=f"Edge Detection Frame {i+1}", channels="BGR", use_column_width=True)

        else:
            st.warning("Video recording has not started yet or lasted less than 10 seconds.")

# Financial Computation Section
elif option == 'Financial Computation':
    st.header('Financial Computation')

    # Input fields for financial parameters
    S0 = st.number_input("Stock Price Today (S0)", min_value=0.0, value=90.0)
    K = st.number_input("Strike Price (K)", min_value=0.0, value=100.0)
    T = st.number_input("Time to Expiry (T in years)", min_value=0.0, value=2.0)
    r = st.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.02)
    sigma = st.number_input("Volatility (σ)", min_value=0.0, value=0.3)
    N = st.number_input("Number of Simulations (N)", min_value=100, value=100000)

    if st.button('Compute Option Price'):
        # Perform the Monte Carlo simulation
        price = monte_carlo_simulation(S0, K, T, r, sigma, N)
        st.write(f"Monte Carlo Estimated Option Price (CPU Parallel): {price:.4f}")

        # Simulate stock prices
        simulated_prices = np.random.randn(N) * sigma * np.sqrt(T) + (r - 0.5 * sigma**2) * T
        ST = S0 * np.exp(simulated_prices)

        # Plot and display the histogram of simulated stock prices
        plot_histogram(ST, S0, K)

# Image Processing Section
elif option == 'Image Processing':
    st.header('Image Processing')

    # Upload image for processing
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        img_bgr, gray_image_rgb, edge_image_rgb, gray_image, edge_image = process_image(uploaded_image)

        # Display images side by side
        st.image([img_bgr, gray_image_rgb, edge_image_rgb], caption=["Original Image", "Grayscale Image", "Edge Detection"], width=500)
        
        # Display histograms
        st.subheader("Grayscale Image Histogram")
        fig, ax = plt.subplots()
        ax.hist(gray_image.ravel(), bins=256, color='gray', alpha=0.7)
        ax.set_title('Grayscale Image Histogram')
        st.pyplot(fig)
        
        st.subheader("Edge Detection Histogram")
        fig, ax = plt.subplots()
        ax.hist(edge_image.ravel(), bins=256, color='black', alpha=0.7)
        ax.set_title('Edge Detection Histogram')
        st.pyplot(fig)

        # Display the original image as a large view
        st.image(img_bgr, caption="Original Image (Large View)", use_column_width=True)
    else:
        st.warning("Please upload an image.")
