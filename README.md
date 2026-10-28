# Autonomous Perception Pipeline

## Overview
This project implements a multi-stage Advanced Driver Assistance System (ADAS) perception pipeline. It integrates deep learning object tracking (YOLOv8 + ByteTrack), classical computer vision lane detection (OpenCV), and spatial depth estimation (MiDaS).

## Architecture
1. **Dynamic Object Tracking:** YOLOv8 + ByteTrack (Ego-vehicle filtered)
2. **Static Lane Detection:** OpenCV (Canny Edge + Hough Transform with vanishing-point ROI)
3. **Depth Estimation:** MiDaS Small (Spatial heat mapping)

## Engineering Outcomes & Insights
- **Spatial Guardrails for Deep Learning:** YOLOv8 successfully tracked vehicles but initially classified the ego-vehicle's hood as a dynamic object. A custom bounding-box interception loop was engineered to drop detections spanning the bottom 15% of the frame, proving that raw model outputs require strict spatial post-processing in ADAS environments.
- **The Fragility of Classical CV:** The OpenCV lane detection module required highly aggressive, hardcoded Region of Interest (ROI) trapezoids to function. Without cutting off the bottom 40% of the frame, the Canny edge detector hallucinated lanes on windshield wipers and dashboard reflections. 
- **Architectural Conclusion:** The pipeline empirically proves that while classical OpenCV is sufficient for strict ego-lane tracking under ideal conditions, its vulnerability to geometric shifts (like drawing lanes on the bumpers of passing cars) makes it unfit for Level 4/5 autonomy. Production-grade multi-lane detection necessitates deep learning semantic segmentation.

## Execution Instructions
Do not install dependencies globally. Create and activate a virtual environment before execution.

```bash
# 1. Create the virtual environment
python -m venv venv

# 2. Activate the environment
# For Mac/Linux:
source venv/bin/activate
# For Windows:
# venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Execute the pipeline
python main.py