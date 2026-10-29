import cv2
import torch
import numpy as np

def generate_depth_map(video_path, frame_number, output_path):
    device = torch.device("cpu")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    cap = cv2.VideoCapture(video_path)
    
    current_frame = 0
    ret = True
    while current_frame <= frame_number and ret:
        ret, frame = cap.read()
        current_frame += 1

    cap.release()

    if not ret or frame is None:
        raise ValueError(f"CRITICAL: Video ended before reaching frame {frame_number}")

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    
    # 1. Isolate the active area (ignoring the dashboard)
    h, w = depth_map.shape
    crop_h = int(h * 0.65)
    active_area = depth_map[:crop_h, :]
    
    # 2. Clip outlier values globally
    min_val = np.min(active_area)
    max_val = np.max(active_area)
    depth_map_clipped = np.clip(depth_map, min_val, max_val)
    
    # 3. Convert to 8-bit integer scale
    depth_map_8u = cv2.normalize(
        depth_map_clipped, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    
    # 4. Apply CLAHE to force local contrast on the edges
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    depth_map_eq = clahe.apply(depth_map_8u)
    
    # 5. Apply Colormap
    depth_colormap = cv2.applyColorMap(depth_map_eq, cv2.COLORMAP_INFERNO)
    cv2.imwrite(output_path, depth_colormap)