import cv2
import torch
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def generate_depth_map(video_path: str, frame_number: int, output_path: str) -> None:
    try:
        logger.info("Downloading/Loading MiDaS depth estimation model...")
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        
        # Dynamic hardware detection guardrail
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.info(f"Mounting MiDaS model to hardware device: {device}")
        midas.to(device)
        midas.eval() 
        
        transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    except Exception as e:
        logger.error(f"Failed to load MiDaS model or transforms: {e}")
        raise

    logger.info(f"Extracting frame {frame_number} from video source...")
    cap = cv2.VideoCapture(video_path)
    
    # Guardrail: Check if OpenCV actually loaded the file
    if not cap.isOpened():
        logger.error(f"OpenCV failed to open the video file at {video_path}.")
        raise ValueError("Invalid video path or corrupted file.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        logger.error(f"Failed to extract frame {frame_number}. It may exceed the video length.")
        raise ValueError("Frame extraction failed.")

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device) 

    logger.info("Calculating spatial depth tensor...")
    try:
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=img.shape[:2], 
                mode="bicubic", align_corners=False
            ).squeeze()

        # Safely move tensor back to CPU for matplotlib saving
        output = prediction.cpu().numpy() 
        plt.imsave(output_path, output, cmap='inferno')
        logger.info(f"Depth map successfully rendered and saved to {output_path}")
    except Exception as e:
        logger.error(f"Depth estimation inference crashed: {e}")
        raise