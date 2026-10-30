```mermaid
graph TD
    A[Input: Raw Video<br/>dashcam.mp4] --> B(cv2.VideoCapture)

    %% Video Processing Pipeline (Dynamic)
    B --> C{YOLOv8 Inference}
    C --> D[ByteTrack ID Assignment]
    D --> E[Spatial Bounding Box Filter<br/>Drops bottom 15% Ego-hood]
    
    B --> F{OpenCV Processing}
    F --> G[ROI Trapezoid Mask<br/>Drops bottom 40%]
    G --> H[Canny Edge + Hough Lines]

    E --> I(main.py: Frame Annotation)
    H --> I
    I --> J[Output: final_dashcam.mp4]

    %% Depth Estimation Pipeline (Static)
    B --> K[Extract Frame 630<br/>Brute-force iteration]
    K --> L{MiDaS Small Tensor}
    L --> M[Crop Dashboard<br/>Top 65% of frame]
    M --> N[CLAHE Normalization<br/>Grid size: 8x8]
    N --> O[Apply OpenCV ColorMap]
    O --> P[Output: depth_map.png]

    classDef input fill:#2d3748,stroke:#a0aec0,stroke-width:2px,color:#fff;
    classDef process fill:#4a5568,stroke:#cbd5e0,stroke-width:1px,color:#fff;
    classDef output fill:#276749,stroke:#9ae6b4,stroke-width:2px,color:#fff;
    
    class A,J,P input;
    class B,C,D,E,F,G,H,I,K,L,M,N,O process;