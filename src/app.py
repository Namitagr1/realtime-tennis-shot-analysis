import streamlit as st
import os
import cv2

# --- Sidebar ---
st.sidebar.title("🎾 Realtime Tennis Shot Analysis")
data_path = "/app/data"

video_files = [f for f in os.listdir(data_path) if f.lower().endswith((".mp4", ".mov", ".avi"))]
selected_video = st.sidebar.selectbox("Choose a video", video_files)

run_pipeline = st.sidebar.button("Run Analysis")

st.title("🎥 Realtime Tennis Shot Analysis")
st.write("Select a tennis video from the data folder.")

# --- Video Player ---
if selected_video:
    video_path = os.path.join(data_path, selected_video)
    st.video(video_path)

# --- Analysis ---
if run_pipeline and selected_video:
    st.subheader("Analysis Results")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video.")
    else:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        st.write(f"Frames: {frame_count}, FPS: {fps:.1f}")

        ret, frame = cap.read()
        if ret:
            h, w, _ = frame.shape
            # placeholder overlay
            cv2.circle(frame, (w // 2, h // 2), 20, (0, 255, 0), 3)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="Sample overlay (ball tracking demo)", use_column_width=True)
        cap.release()
