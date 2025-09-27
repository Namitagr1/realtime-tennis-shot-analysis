import streamlit as st
import os
import cv2
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt
from ultralytics import YOLO
import mediapipe as mp

# --- Sidebar ---
st.sidebar.title("🎾 Tennis Shot Analysis (Player + Ball)")

# Adjusted paths relative to src/
data_path = os.path.join(os.path.dirname(__file__), "../data")
model_path = os.path.join(os.path.dirname(__file__), "../models/yolov8s.pt")

video_files = [f for f in os.listdir(data_path) if f.lower().endswith((".mp4", ".mov", ".avi"))]
selected_video = st.sidebar.selectbox("Choose a video", video_files)

# Instead of court width, front-on video uses ball diameter in pixels
pixel_ball_diameter = st.sidebar.number_input(
    "Ball diameter in pixels (pause video and measure width of ball)", min_value=2, value=6
)
meters_per_pixel = 0.067 / pixel_ball_diameter  # tennis ball diameter = 6.7 cm

run_pipeline = st.sidebar.button("Run Analysis")

st.title("🎥 Tennis Shot Analysis (Front-On, Player & Ball)")
st.write("Tracks ball speed/spin and highlights player's arm swing for any shot.")

# --- MediaPipe pose setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Main pipeline ---
if run_pipeline and selected_video:
    if not os.path.exists(model_path):
        st.error(f"YOLOv8 model not found at {model_path}. Place yolov8s.pt there.")
    else:
        model = YOLO(model_path)

        video_path = os.path.join(data_path, selected_video)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not open video.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            st.write(f"Frames: {frame_count}, FPS: {fps:.1f}")

            # Streamlit placeholders
            frame_placeholder = st.empty()
            stats_placeholder = st.empty()
            graph_placeholder = st.empty()
            summary_placeholder = st.empty()

            # Track ball and arm metrics
            prev_positions = deque(maxlen=5)
            speed_history_mph = []
            spin_history_rpm = []
            arm_angle_history = []

            max_speed_mph = 0.0
            max_spin_rpm = 0.0
            max_arm_angle = 0.0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # --- Ball detection ---
                results = model.predict(frame, verbose=False, conf=0.5, device='cpu')
                ball_speed_mph = 0.0
                angle_deg = 0.0
                spin_rpm = 0.0
                predicted_landing = None

                if results and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    scores = results[0].boxes.conf.cpu().numpy()
                    ball_indices = [i for i, c in enumerate(classes) if int(c) == 32]
                    if ball_indices:
                        idx = max(ball_indices, key=lambda i: scores[i])
                        x1, y1, x2, y2 = boxes[idx]
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        prev_positions.append((cx, cy))
                        radius = int(max(x2 - x1, y2 - y1)/2)
                        cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 2)

                        for i in range(1, len(prev_positions)):
                            cv2.line(frame, prev_positions[i-1], prev_positions[i], (0, 255, 255), 2)

                        # Speed
                        if len(prev_positions) >= 2:
                            (x1p, y1p), (x2p, y2p) = prev_positions[-2], prev_positions[-1]
                            distance_m = np.sqrt((x2p - x1p)**2 + (y2p - y1p)**2) * meters_per_pixel
                            ball_speed_mps = distance_m * fps
                            ball_speed_mph = ball_speed_mps * 2.23694
                            speed_history_mph.append(ball_speed_mph)
                            max_speed_mph = max(max_speed_mph, ball_speed_mph)

                            dx = x2p - x1p
                            dy = y2p - y1p
                            if dx != 0:
                                angle_deg = np.degrees(np.arctan2(-dy, dx))

                            # Spin RPM
                            if len(prev_positions) >= 3:
                                x0, y0 = prev_positions[-3]
                                curve_m = ((x2p - x1p) - (x1p - x0)) * meters_per_pixel
                                spin_rpm = abs(curve_m / 0.026) * fps * 60
                                spin_history_rpm.append(spin_rpm)
                                max_spin_rpm = max(max_spin_rpm, spin_rpm)

                            # Predict landing (linear)
                            if dx != 0:
                                slope = dy/dx
                                x_target = frame.shape[1]
                                y_target = int(cy + slope * (x_target - cx))
                                predicted_landing = (x_target, y_target)
                                cv2.circle(frame, predicted_landing, 5, (0,0,255), -1)
                                cv2.line(frame, (cx, cy), predicted_landing, (0,0,255),1)

                # --- Player pose detection ---
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(frame_rgb)

                arm_angle = 0.0
                if pose_results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )

                    lm = pose_results.pose_landmarks.landmark
                    shoulder = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                                         lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]])
                    elbow = np.array([lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame.shape[1],
                                      lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame.shape[0]])
                    wrist = np.array([lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * frame.shape[1],
                                      lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame.shape[0]])

                    ba = shoulder - elbow
                    bc = wrist - elbow
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    arm_angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
                    arm_angle_history.append(arm_angle)
                    max_arm_angle = max(max_arm_angle, arm_angle)

                # Overlay stats
                cv2.putText(frame, f"Speed: {ball_speed_mph:.1f} mph", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                cv2.putText(frame, f"Angle: {angle_deg:.1f} deg", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
                cv2.putText(frame, f"Spin: {spin_rpm:.0f} RPM", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)
                cv2.putText(frame, f"Arm Angle: {arm_angle:.1f}°", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)

                frame_placeholder.image(frame, use_container_width=True)

                stats_placeholder.markdown(
                    f"**Ball Speed:** {ball_speed_mph:.1f} mph  \n"
                    f"**Shot Angle:** {angle_deg:.1f}°  \n"
                    f"**Spin:** {spin_rpm:.0f} RPM  \n"
                    f"**Arm Angle:** {arm_angle:.1f}°  \n"
                    + (f"**Predicted Landing:** ({predicted_landing[0]}, {predicted_landing[1]})" if predicted_landing else "")
                )

                if speed_history_mph:
                    plt.figure(figsize=(6,2))
                    plt.plot(speed_history_mph, color='red', linewidth=2)
                    plt.title("Ball Speed Over Time (mph)")
                    plt.xlabel("Frame")
                    plt.ylabel("Speed (mph)")
                    plt.grid(True)
                    plt.tight_layout()
                    graph_placeholder.pyplot(plt)
                    plt.close()

                summary_placeholder.markdown(
                    f"**Max Speed:** {max_speed_mph:.1f} mph  \n"
                    f"**Max Spin:** {max_spin_rpm:.0f} RPM  \n"
                    f"**Max Arm Angle:** {max_arm_angle:.1f}°"
                )

                time.sleep(1.0 / fps)

            cap.release()
            st.success("Video analysis finished!")
