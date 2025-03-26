import streamlit as st
import cv2
import time
from numba import njit, prange
import numpy as np

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

# Streamlit Video Processing Section
if option == 'Video Processing':
    st.header('Real-Time Video Processing')

    # Start webcam button
    if 'video_processing' not in st.session_state:
        st.session_state.video_processing = False
        st.session_state.cap = None  # Initialize webcam capture object

    start_button = st.button('Start Video Processing')

    # Start Video Processing
    if start_button:
        st.session_state.video_processing = True
        if st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(0)

        if not st.session_state.cap.isOpened():
            st.error("Error: Could not open webcam.")
            exit()

        st.experimental_rerun()  # Trigger a rerun to start the video processing loop

    # If video processing is active, start the webcam
    if st.session_state.video_processing:
        # Create a placeholder for displaying the processed frames
        video_placeholder = st.empty()

        # Placeholder for FPS count
        fps_placeholder = st.empty()

        frame_count = 0
        start_time = time.time()

        # Record video for 10 seconds
        while st.session_state.video_processing:  # Process video while active
            elapsed_time = time.time() - start_time
            if elapsed_time > 10:
                st.session_state.video_processing = False  # Stop recording after 10 seconds
                break

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
            fps = frame_count / elapsed_time
            fps_placeholder.text(f"FPS: {fps:.2f}")

            # Display the processed video in the Streamlit app
            video_placeholder.image(
                [frame_bgr, gray_frame_bgr, edge_frame_bgr], 
                caption=["Original", "Grayscale", "Edge Detection"], 
                channels="RGB", 
                use_column_width=True
            )

        st.session_state.cap.release()  # Release the webcam after processing is stopped
        st.info("Video processing finished. Processed frames have been displayed.")
    else:
        st.info("Click 'Start Video Processing' to begin.")
