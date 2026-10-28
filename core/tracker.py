import logging
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

def run_tracking(video_path: str, output_dir: str) -> None:
    try:
        logger.info("Initializing YOLOv8s PyTorch model...")
        model = YOLO('yolov8s.pt')
    except Exception as e:
        logger.error(f"Failed to load YOLO model. Check internet or ultralytics installation: {e}")
        raise

    try:
        # Dynamic hardware routing for maximum latency reduction
        device = "0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Executing tracking pipeline on device: {device} (Filtering for Vehicles: COCO 2, 3, 5, 7)...")
        
        results = model.track(
            source=video_path, 
            tracker='bytetrack.yaml', 
            classes=[2, 3, 5, 7], 
            device=device,
            save=True,
            project=output_dir,
            name='tracking_results',
            exist_ok=True
        )
        logger.info("YOLOv8 tracking completed successfully.")
    except Exception as e:
        logger.error(f"Tracking inference failed during execution: {e}")
        raise