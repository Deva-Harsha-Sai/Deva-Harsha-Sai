import streamlit as st
import cv2
import numpy as np
import time
from numba import njit, prange
import matplotlib.pyplot as plt
import multiprocessing
from io import BytesIO
from PIL import Image

# Streamlit App Title
st.title('CUDA Python Task Interface')

# Sidebar for Navigation
option = st.sidebar.selectbox(
    'Select a Project',
    ['Homepage', 'Video Processing', 'Financial Computation', 'Image Processing']
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

    st.subheader("Why Use CUDA Python?")
    st.write("""
    Using CUDA Python can significantly speed up computations by utilizing the power of GPUs. GPUs are highly parallel and can process many tasks simultaneously, making them well-suited for operations like matrix multiplications, convolutions, and other mathematical computations.
    This makes CUDA Python ideal for tasks that require heavy computation, such as:
    - Image and video processing
    - Financial computations
    - Machine learning
    - Simulation-based tasks
    """)

    # Footer
    st.markdown("""
    ---
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

# Financial Computation Functions (Monte Carlo Simulation for Option Pricing)
def monte_carlo_simulation(S0, K, T, r, sigma, N):
    np.random.seed(42)
    dt = T
    discount_factor = np.exp(-r * T)
    
    payoffs = np.zeros(N)
    for i in prange(N):
        Z = np.random.randn()
        ST = S0 * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        payoffs[i] = max(ST - K, 0)  # Call option payoff

    return discount_factor * np.mean(payoffs)

def plot_histogram(simulated_prices, S0, K):
    plt.figure(figsize=(10, 5))
    plt.hist(simulated_prices, bins=50, color='blue', alpha=0.6, edgecolor='black')
    plt.axvline(x=K, color='red', linestyle="--", label="Strike Price (K)")
    plt.xlabel("Simulated Stock Price at Maturity")
    plt.ylabel("Frequency")
    plt.title("Monte Carlo Simulated Stock Prices")
    plt.legend()
    st.pyplot(plt)

# Image Processing Functions
def process_image(uploaded_image):
    """
    This function processes the uploaded image (e.g., converts to grayscale).
    """
    img_array = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Sobel Edge Detection
    edge_image = sobel_edge_detection(gray_image)

    # Convert images to RGB format for Streamlit display
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_image_rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    edge_image_rgb = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2RGB)

    return img_bgr, gray_image_rgb, edge_image_rgb, gray_image, edge_image

# Video Processing Section
if option == 'Video Processing':
    st.header('Real-Time Video Processing')

    # Start webcam button
    if 'video_processing' not in st.session_state:
        st.session_state.video_processing = False
        st.session_state.cap = None  # Initialize webcam capture object

    start_button = st.button('Start Video Processing')
    stop_button = st.button('Stop Video Processing')

    # Start Video Processing
    if start_button:
        st.session_state.video_processing = True
        if st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(0)

        if not st.session_state.cap.isOpened():
            st.error("Error: Could not open webcam.")
            exit()

        st.experimental_rerun()  # Trigger a rerun to start the video processing loop

    # Stop Video Processing
    if stop_button:
        st.session_state.video_processing = False
        if st.session_state.cap:
            st.session_state.cap.release()  # Release the webcam
        st.experimental_rerun()  # Trigger a rerun to stop the video processing loop

    # If video processing is active, start the webcam
    if st.session_state.video_processing:
        # Create a placeholder for displaying the processed frames
        video_placeholder = st.empty()

        # Placeholder for FPS count
        fps_placeholder = st.empty()

        frame_count = 0
        start_time = time.time()

        while st.session_state.video_processing:  # Process video while active
            ret, frame = st.session_state.cap.read()
            if not ret:
                break

            # Grayscale conversion
            gray_frame = grayscale_numba(frame)

            # Edge Detection (Sobel Filter)
            edge_frame = sobel_edge_detection(gray_frame)

            # Convert frames to 3-channel images for display
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Original Frame
            gray_frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)  # Grayscale Frame
            edge_frame_bgr = cv2.cvtColor(edge_frame, cv2.COLOR_GRAY2RGB)  # Edge Detected Frame

            # FPS calculation
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            fps_placeholder.text(f"FPS: {fps:.2f}")

            # Display the processed video in the Streamlit app
            video_placeholder.image(
                [frame_bgr, gray_frame_bgr, edge_frame_bgr], 
                caption=["Original", "Grayscale", "Edge Detection"], 
                channels="RGB", 
                use_column_width=True
            )

            # Allow the user to stop the stream
            if not st.session_state.video_processing:
                break

        st.session_state.cap.release()  # Release the webcam after processing is stopped
    else:
        st.info("Click 'Start Video Processing' to begin.")

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
