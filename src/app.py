import streamlit as st
import os
import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from ultralytics import YOLO
import mediapipe as mp

# ==============================
# Sidebar Config
# ==============================
st.sidebar.title("🎾 Tennis Shot Analysis (Player + Ball)")

# Paths relative to src/
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "../data")
MODEL_DIR = os.path.join(BASE_DIR, "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "yolov8s.pt")

# Video selector
video_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".mp4", ".mov", ".avi"))]
selected_video = st.sidebar.selectbox("Choose a video", video_files)

# Calibration: ball diameter in pixels
pixel_ball_diameter = 31

meters_per_pixel = 0.067 / pixel_ball_diameter

run_pipeline = st.sidebar.button("Run Analysis")

# ==============================
# Page Title
# ==============================
st.title("🎥 Tennis Shot Analysis (Front-On View)")
st.write("Tracks ball speed, spin, and predicts landing. Highlights player’s arm swing angles.")

# ==============================
# MediaPipe Pose Setup
# ==============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ==============================
# Main Pipeline
# ==============================
if run_pipeline and selected_video:
    if not os.path.exists(MODEL_PATH):
        st.error(f"YOLOv8 model not found at {MODEL_PATH}. Place yolov8s.pt in models/")
    else:
        model = YOLO(MODEL_PATH)

        video_path = os.path.join(DATA_DIR, selected_video)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("❌ Could not open video.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            st.write(f"Video loaded: {frame_count} frames at {fps:.1f} FPS ({width}x{height})")

            # Streamlit placeholders
            frame_placeholder = st.empty()
            stats_placeholder = st.empty()
            graph_placeholder = st.empty()
            summary_placeholder = st.empty()
            progress_bar = st.progress(0)

            # Histories
            prev_positions = deque(maxlen=5)
            speed_history = deque(maxlen=2000)
            spin_history = deque(maxlen=2000)
            arm_angle_history = deque(maxlen=2000)

            max_speed = 0.0
            max_spin = 0.0
            max_arm_angle = 0.0

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                # -------------------------
                # Ball Detection
                # -------------------------
                results = model.predict(frame, verbose=False, conf=0.5, device="cpu")
                ball_speed_mph, spin_rpm, angle_deg = 0.0, 0.0, 0.0
                predicted_landing = None

                if results and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    scores = results[0].boxes.conf.cpu().numpy()

                    ball_indices = [i for i, c in enumerate(classes) if int(c) == 32]  # tennis ball class
                    if ball_indices:
                        idx = max(ball_indices, key=lambda i: scores[i])
                        x1, y1, x2, y2 = boxes[idx]
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                        prev_positions.append((cx, cy))
                        radius = int(max(x2 - x1, y2 - y1) / 2)
                        cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 2)

                        # Draw path
                        for i in range(1, len(prev_positions)):
                            cv2.line(frame, prev_positions[i - 1], prev_positions[i], (0, 255, 255), 2)

                        # Speed
                        if len(prev_positions) >= 2:
                            (x1p, y1p), (x2p, y2p) = prev_positions[-2], prev_positions[-1]
                            dist_m = np.sqrt((x2p - x1p) ** 2 + (y2p - y1p) ** 2) * meters_per_pixel
                            ball_speed_mps = dist_m * fps
                            ball_speed_mph = ball_speed_mps * 2.23694
                            speed_history.append(ball_speed_mph)
                            max_speed = max(max_speed, ball_speed_mph)

                            dx, dy = (x2p - x1p), (y2p - y1p)
                            if dx != 0:
                                angle_deg = np.degrees(np.arctan2(-dy, dx))

                            # Spin
                            if len(prev_positions) >= 3:
                                x0, y0 = prev_positions[-3]
                                curve_m = ((x2p - x1p) - (x1p - x0)) * meters_per_pixel
                                spin_rpm = abs(curve_m / 0.026) * fps * 60
                                spin_history.append(spin_rpm)
                                max_spin = max(max_spin, spin_rpm)

                            # Landing Prediction
                            if dx != 0:
                                slope = dy / dx
                                x_target = frame.shape[1]
                                y_target = int(cy + slope * (x_target - cx))
                                predicted_landing = (x_target, y_target)
                                if 0 <= y_target < frame.shape[0]:
                                    cv2.circle(frame, predicted_landing, 5, (0, 0, 255), -1)
                                    cv2.line(frame, (cx, cy), predicted_landing, (0, 0, 255), 1)

                # -------------------------
                # Pose Detection (Player)
                # -------------------------
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(frame_rgb)

                arm_angle = 0.0
                if pose_results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )

                    lm = pose_results.pose_landmarks.landmark
                    try:
                        shoulder = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                                             lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]])
                        elbow = np.array([lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame.shape[1],
                                          lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame.shape[0]])
                        wrist = np.array([lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * frame.shape[1],
                                          lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame.shape[0]])

                        ba, bc = shoulder - elbow, wrist - elbow
                        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                        arm_angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
                        arm_angle_history.append(arm_angle)
                        max_arm_angle = max(max_arm_angle, arm_angle)
                    except Exception:
                        pass

                # -------------------------
                # Overlay Stats
                # -------------------------
                cv2.putText(frame, f"Speed: {ball_speed_mph:.1f} mph", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Angle: {angle_deg:.1f} deg", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"Spin: {spin_rpm:.0f} RPM", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Arm Angle: {arm_angle:.1f}°", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Streamlit Updates
                frame_placeholder.image(frame, channels="BGR", use_container_width=True)

                stats_placeholder.markdown(
                    f"**Ball Speed:** {ball_speed_mph:.1f} mph  \n"
                    f"**Shot Angle:** {angle_deg:.1f}°  \n"
                    f"**Spin:** {spin_rpm:.0f} RPM  \n"
                    f"**Arm Angle:** {arm_angle:.1f}°  \n"
                    + (f"**Predicted Landing:** {predicted_landing}" if predicted_landing else "")
                )

                # Plot graphs
                if speed_history:
                    plt.figure(figsize=(6, 2))
                    plt.plot(speed_history, color="red", linewidth=2, label="Speed (mph)")
                    if spin_history:
                        plt.plot(spin_history, color="blue", linewidth=1, alpha=0.7, label="Spin (RPM)")
                    plt.legend()
                    plt.title("Ball Metrics Over Time")
                    plt.xlabel("Frame")
                    plt.grid(True)
                    plt.tight_layout()
                    graph_placeholder.pyplot(plt)
                    plt.close()

                # Progress
                progress_bar.progress(frame_idx / frame_count)

            cap.release()
            summary_placeholder.markdown(
                f"### 📊 Summary  \n"
                f"- **Max Speed:** {max_speed:.1f} mph  \n"
                f"- **Max Spin:** {max_spin:.0f} RPM  \n"
                f"- **Max Arm Angle:** {max_arm_angle:.1f}°"
            )
            st.success("✅ Video analysis finished!")
