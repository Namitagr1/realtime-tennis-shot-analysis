import streamlit as st
import os
import cv2
import time

# --- Sidebar ---
st.sidebar.title("🎾 Realtime Tennis Shot Analysis")
data_path = "/app/data"

video_files = [f for f in os.listdir(data_path) if f.lower().endswith((".mp4", ".mov", ".avi"))]
selected_video = st.sidebar.selectbox("Choose a video", video_files)

run_pipeline = st.sidebar.button("Run Analysis")

st.title("🎥 Realtime Tennis Shot Analysis")
st.write("Select a tennis video from the data folder and run analysis.")

# --- Video Player / Real-time Analysis ---
if run_pipeline and selected_video:
    video_path = os.path.join(data_path, selected_video)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video.")
    else:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        st.write(f"Frames: {frame_count}, FPS: {fps:.1f}")

        # Create a placeholder for Streamlit to update frames
        frame_placeholder = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # --- Example: overlay a fake ball ---
            h, w, _ = frame.shape
            center = (w // 2, h // 2)
            cv2.circle(frame, center, 20, (0, 255, 0), 3)

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Update frame in Streamlit
            frame_placeholder.image(frame_rgb, use_container_width=True)

            # Sleep to match original video FPS
            time.sleep(1.0 / fps)

        cap.release()
        st.success("Video analysis finished!")
