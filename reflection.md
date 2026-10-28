# Individual Engineering Reflection

**Author:** Manoj K. S.  
**Role:** Perception & Software Pipeline Engineer  
**Project:** Multi-Stage Autonomous Perception Pipeline (ADAS)

---

## 1. Scope of Ownership
I owned the end-to-end integration of the perception pipeline, coordinating three distinct computer vision and deep learning components into an automated, sequential system (`main.py`):
- **Object Detection & Tracking:** Deployed YOLOv8 coupled with ByteTrack for persistent vehicle identification.
- **Monocular Depth Estimation:** Loaded and executed MiDaS Small via PyTorch Hub to output spatial heatmaps.
- **Classical Lane Detection:** Engineered an OpenCV pipeline utilizing Gaussian filtering, Canny edge detection, and Probabilistic Hough Line Transforms.
- **Pipeline Hardening & Guardrails:** Designed spatial intercept loops, ROI geometric constraints, structured logging, and dependency isolation.

---

## 2. Key Engineering Challenges & Solutions

### A. Ego-Vehicle False Positives in Deep Learning
- **The Problem:** YOLOv8 identified the host car's hood as an obstacle because it occupied the bottom 35–40% of the field of view, assigning dynamic tracking IDs to the ego-vehicle.
- **The Fix:** Rather than retraining weights, I implemented a programmatic spatial intercept filter directly on the bounding-box tensors. By filtering out any detection where the lower boundary ($y_2$) exceeded 85% of frame height while spanning over 50% of the total frame width, the hood was eliminated from tracking calculations with zero additional inference latency.

### B. Classical Edge Detection Fragility
- **The Problem:** The Canny edge detector routinely hallucinated lanes across windshield wipers, dashboard reflections, and passing vehicle contours due to sharp local gradient contrast.
- **The Fix:** I constricted the Region of Interest (ROI) into a sharp trapezoidal corridor, cutting off the bottom 40% of the frame to isolate the camera hood entirely and tuning the Hough line slope thresholds ($|\text{slope}| \ge 0.4$) to reject lateral visual artifacts.

---

## 3. Technical Conclusions & Key Learnings
1. **Classical CV vs. Deep Learning:** While classical OpenCV algorithms are lightweight and computationally cheap for controlled highway ego-lanes, they lack semantic awareness. High-contrast vehicle bumpers crossing into the ROI inevitably generate false-positive lane lines.
2. **ADAS Redundancy:** Production-grade Level 4/5 autonomous vehicles cannot rely on heuristic edge-finding. Next iterations require migrating static lane tracking to deep semantic segmentation networks (such as Ultra-Fast-Lane-Detection or SegNet) while running parallel multi-object tracking.