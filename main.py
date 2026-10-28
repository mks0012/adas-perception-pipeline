import os
import sys
import logging
import cv2
from ultralytics import YOLO
from core.depth import generate_depth_map
from core.lane import detect_lanes

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    video_source = "dashcam.mp4"
    output_dir = "output"
    frame_to_extract = 450
    
    yolo_output_video = os.path.join(output_dir, "yolo_filtered.mp4")
    final_integrated_video = os.path.join(output_dir, "final_dashcam.mp4")
    depth_output_path = os.path.join(output_dir, "depth_map.png")

    logger.info("Initializing Nexvia Perception Pipeline...")

    if not os.path.exists(video_source):
        logger.error(f"CRITICAL: Input video '{video_source}' not found in the root directory.")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ==========================================
    # Phase 1: YOLO Tracking with Pro-Grade UI
    # ==========================================
    logger.info("Starting Phase 1: YOLOv8 Tracking...")
    try:
        model = YOLO('yolov8n.pt') 
        cap = cv2.VideoCapture(video_source)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(yolo_output_video, fourcc, fps, (width, height))
        
        results = model.track(source=video_source, stream=True, persist=True)
        
        for r in results:
            frame = r.orig_img.copy()
            boxes = r.boxes
            
            if boxes is not None and boxes.id is not None:
                for box, track_id in zip(boxes, boxes.id):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Spatial Filter: Drop the ego-vehicle hood
                    box_width = x2 - x1
                    if y2 > height * 0.85 and box_width > width * 0.50:
                        continue  
                        
                    # Pro-Grade UI Drawing
                    color = (255, 255, 0) # High-visibility Cyan
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    label = f"ID:{int(track_id)} {model.names[cls].upper()} {conf:.2f}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Solid background block for text
                    cv2.rectangle(frame, (int(x1), int(y1) - 25), (int(x1) + w, int(y1)), color, -1)
                    cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
            out.write(frame)
            
        cap.release()
        out.release()
        logger.info(f"Phase 1 Complete: Filtered tracking video saved to {yolo_output_video}")
    except Exception as e:
        logger.error(f"Phase 1 Failed: {str(e)}")
        sys.exit(1)

    # ==========================================
    # Phase 2: MiDaS Depth Estimation
    # ==========================================
    try:
        logger.info(f"Starting Phase 2: MiDaS Depth Estimation on frame {frame_to_extract}")
        generate_depth_map(video_source, frame_to_extract, depth_output_path)
        logger.info(f"Phase 2 Complete: Depth map saved to {depth_output_path}")
    except Exception as e:
        logger.error(f"Phase 2 Failed: {str(e)}")
        sys.exit(1)

    # ==========================================
    # Phase 3: OpenCV Lane Detection Integration
    # ==========================================
    logger.info("Starting Phase 3: Integrating OpenCV Lane Detection...")
    if os.path.exists(yolo_output_video):
        detect_lanes(yolo_output_video, final_integrated_video)
        logger.info(f"Phase 3 Complete: Fully integrated video saved to {final_integrated_video}")
    else:
        logger.error("Could not locate the YOLO output video to run lane detection.")
        sys.exit(1)

    logger.info("Pipeline execution successfully terminated. Review 'output' directory.")

if __name__ == "__main__":
    main()