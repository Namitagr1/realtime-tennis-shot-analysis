# RT-Tennis: Real Time Tennis Analysis

## Setup
```bash
git clone https://github.com/Namitagr1/realtime-tennis-shot-analysis.git
cd realtime-tennis-shot-analysis
docker-compose up --build
```

## Inspiration
Tennis broadcasts provide fans and analysts with a rich visual experience, but quantitative insights like ball speed, spin, and shot mechanics are often locked behind expensive proprietary systems. We wanted to create an open, accessible tool that leverages computer vision and modern AI techniques to bring professional-grade analysis to anyone with match footage.

## What it does
Real Time Tennis Analysis is a multi-module platform that:
- Tracks the tennis ball in video and estimates speed, spin, and trajectory in real time.
- Detects player pose and highlights arm mechanics across different shot types.
- Integrates large historical ATP datasets to surface performance trends, player comparisons, and surface-specific breakdowns.
- Provides AI-powered coaching tips using Google’s Gemini API, contextualized by the type of shot being analyzed.

## How we built it
We used **YOLOv8** for object detection (specifically to localize the tennis ball) and **MediaPipe Pose** for tracking player joints and arm angles. To make noisy detections stable, we integrated **Kalman filtering** and interpolation to handle frames where the ball leaves the camera view. Historical ATP match CSVs were ingested with **pandas** and visualized via **Plotly dashboards**, providing both player-specific and head-to-head trend analyses. A Streamlit-based frontend ties these together into a unified multi-page application, containerized with Docker for reproducibility. We also integrated the **Gemini API** to generate contextual coaching insights.  

## Challenges we ran into
- **Ball detection**: The tennis ball is extremely small and often blurred in match footage, making generic object detection models unreliable. We experimented with multiple filters and motion models to stabilize trajectories.
- **Court calibration**: Mapping pixel distances to real-world speeds required homography estimation, which is difficult in broadcast footage with shifting camera angles.
- **Data integration**: Cleaning ATP match datasets across years revealed inconsistent columns and missing values, which we had to standardize before building trend visualizations.
- **System complexity**: Merging real-time CV pipelines with large-scale historical analytics, AI-generated insights, and a clean user interface pushed the limits of both compute and design.

## Accomplishments that we're proud of
- Built a working pipeline that overlays **real-time ball and player metrics** on raw video footage.
- Created an **interactive analytics dashboard** for exploring years of ATP data, including player trends, surface-specific performance, and head-to-head breakdowns.
- Integrated a modern **generative AI API** to provide contextual coaching suggestions alongside quantitative stats.
- Successfully packaged the project in Docker, making it portable and easy to deploy.

## What we learned
We learned how to combine multiple domains—computer vision, sports analytics, and generative AI—into a single coherent system. We also deepened our understanding of the challenges of small-object detection, signal smoothing for noisy trajectories, and how raw sports data needs careful preprocessing to become meaningful for visualization and analysis.

## What's next for Real Time Tennis Analysis
In the future, we want to:
- Train or fine-tune a detection model specifically for tennis footage to improve ball tracking accuracy.
- Improve **physics-based spin and trajectory models** to produce more reliable statistics.
- Extend the analytics dashboard into a **real-time broadcast overlay tool**, allowing fans and coaches to see metrics live.
- Explore integration with wearable data or additional sensor streams for richer multi-modal analysis.
