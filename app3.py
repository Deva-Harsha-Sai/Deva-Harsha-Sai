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
    ['Homepage', 'Image Processing', 'Financial Computation', 'Video Processing']
)

# Homepage Section
if option == 'Homepage':
    st.header('CUDA Python Tasks: Real-Time Video Processing, Financial Computation & Image Processing')
    
    st.subheader("What is CUDA Python?")
    st.write("""CUDA Python allows Python code execution on GPUs using NVIDIA's CUDA platform, accelerating applications in machine learning, image processing, and scientific computing.""")
    
    st.markdown("""---\n**By**: Deva Harsha Sai Nangunuri""")

@njit(parallel=True)
def grayscale_numba(frame):
    gray = np.empty((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    for i in prange(frame.shape[0]):
        for j in prange(frame.shape[1]):
            r, g, b = frame[i, j]
            gray[i, j] = int(0.2989 * r + 0.5870 * g + 0.1140 * b)
    return gray

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

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = grayscale_numba(img)
        edges = sobel_edge_detection(gray)
        return av.VideoFrame.from_ndarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), format="bgr24")

if option == 'Video Processing':
    st.header('Real-Time Video Processing')
    webrtc_streamer(key="video", video_processor_factory=VideoProcessor)

if option == 'Financial Computation':
    st.header('Monte Carlo Option Pricing')
    S0 = st.number_input("Stock Price Today (S0)", min_value=0.0, value=90.0)
    K = st.number_input("Strike Price (K)", min_value=0.0, value=100.0)
    T = st.number_input("Time to Expiry (T in years)", min_value=0.0, value=2.0)
    r = st.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.02)
    sigma = st.number_input("Volatility (σ)", min_value=0.0, value=0.3)
    N = st.number_input("Number of Simulations (N)", min_value=100, value=100000)

    if st.button('Compute Option Price'):
        dt = T / 365
        option_prices = [max(S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.randn()) - K, 0) for _ in range(N)]
        option_price = np.exp(-r * T) * np.mean(option_prices)
        st.write(f"Estimated Option Price: ${option_price:.2f}")

        plt.figure(figsize=(8, 6))
        plt.hist(option_prices, bins=50, color='blue', alpha=0.7)
        plt.axvline(x=S0, color='red', linestyle='--', label=f'Stock Price Today: {S0}')
        plt.axvline(x=K, color='green', linestyle='--', label=f'Strike Price: {K}')
        plt.title('Simulated Stock Prices')
        plt.xlabel('Stock Price')
        plt.ylabel('Frequency')
        plt.legend()
        st.pyplot()

if option == 'Image Processing':
    st.header('Image Processing')
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        img_bgr = np.array(img)
        gray_image = grayscale_numba(img_bgr)
        edge_image = sobel_edge_detection(gray_image)
        gray_image_rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        edge_image_rgb = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2RGB)
        
        st.image([img_bgr, gray_image_rgb, edge_image_rgb], caption=["Original", "Grayscale", "Edges"], width=500)
        
        st.subheader("Grayscale Histogram")
        fig, ax = plt.subplots()
        ax.hist(gray_image.ravel(), bins=256, color='gray', alpha=0.7)
        ax.set_title('Grayscale Histogram')
        st.pyplot(fig)
        
        st.subheader("Edge Detection Histogram")
        fig, ax = plt.subplots()
        ax.hist(edge_image.ravel(), bins=256, color='black', alpha=0.7)
        ax.set_title('Edge Detection Histogram')
        st.pyplot(fig)
    else:
        st.warning("Please upload an image.")
