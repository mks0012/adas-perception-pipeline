# Pipeline Architecture

```mermaid
graph TD
    A[Raw Video Frame: dashcam.mp4] --> B[YOLOv8 + ByteTrack]
    A --> C[OpenCV + Hough Transform]
    A --> D[MiDaS Small]
    
    B --> E[Dynamic Tracking: Spatial Ego-Vehicle Filter]
    C --> F[Static Lane Detection: ROI Trapezoid Mask]
    D --> G[Spatial Depth Map]
    
    E --> H[Integration Script: main.py]
    F --> H
    
    H --> I[Output: final_dashcam.mp4]
    G --> J[Output: depth_map.png]