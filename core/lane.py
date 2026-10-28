import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def detect_lanes(video_path: str, output_path: str) -> None:
    logger.info("Initializing OpenCV Lane Detection module...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        raise ValueError("Video source unavailable.")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # Ultra-Aggressive Trapezoid ROI
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, int(height * 0.60)),                   
            (width, int(height * 0.60)),               
            (int(width * 0.55), int(height * 0.45)),   
            (int(width * 0.45), int(height * 0.45))    
        ]], np.int32)
        
        cv2.fillPoly(mask, polygon, 255)
        cropped_edges = cv2.bitwise_and(edges, mask)
        
        # Bridging gaps heavily for dashed lines
        lines = cv2.HoughLinesP(
            cropped_edges, 1, np.pi/180, 30, 
            np.array([]), minLineLength=20, maxLineGap=100
        )
        
        line_image = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.flatten()
                
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.4:  
                    continue
                    
                # Heavy, production-grade lane marking thickness
                cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 10)
                
        # 1.0 weight on original frame prevents it from dimming, 1.0 on lines makes them vivid
        combo_image = cv2.addWeighted(frame, 1.0, line_image, 1.0, 0)
        out.write(combo_image)

    cap.release()
    out.release()
    logger.info(f"Lane detection completed. Saved to {output_path}")