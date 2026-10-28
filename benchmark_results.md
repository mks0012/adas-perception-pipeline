# Benchmark & Validation Results

## Environment & Hardware Specs
- **Hardware:** Apple M1 MacBook Air (8-core CPU, 7-core GPU, Unified Memory)
- **Runtime Environment:** macOS, Python 3.14 (Virtual Environment)
- **Execution Target:** CPU fallback (`torch.device("cpu")`)

## 1. Latency & Inference Performance
Metrics captured during full pipeline batch execution across 818 frames:

| Pipeline Stage | Module / Model | Input Shape | Latency per Frame | Throughput |
| :--- | :--- | :--- | :--- | :--- |
| Preprocessing | OpenCV / Torch Tensor | 384x640 | 4.0 ms – 7.0 ms | ~140–250 FPS |
| Object Tracking | YOLOv8n + ByteTrack | 384x640 | 147.0 ms – 167.7 ms | ~6–7 FPS |
| Postprocessing | Intercept Filter + NMS | 384x640 | 0.9 ms – 1.1 ms | ~900–1100 FPS |
| Monocular Depth | MiDaS Small (`midas_v21_small_256`) | 256x256 | ~688 ms (frame 450) | ~1.4 FPS |
| Static Lane Marking | Canny + Probabilistic Hough | 384x640 | ~18 ms | ~55 FPS |

## 2. Test Cases & Validation Scenarios

| Test Case | Scenario / Condition | Expected Output | Actual Behavior | Resolution / Engineering Fix |
| :--- | :--- | :--- | :--- | :--- |
| **TC-01: Ego-Vehicle Occlusion** | Car hood takes up bottom 35–40% of the camera frame | No bounding box or ID assigned to ego-vehicle | Raw YOLOv8 classified hood as a tracked vehicle (`id:244 car`) | Implemented spatial threshold filter dropping boxes where `y2 > 0.85 * height` and `width > 0.50 * frame_width`. |
| **TC-02: Windshield Reflection** | Wiper blades and glare create sharp edge contrast | Classical lane module ignores wiper edges | Canny edge detector drew green lane lines across wiper geometry | Elevated ROI trapezoid base to `0.60 * height`, physically isolating camera obstruction. |
| **TC-03: Dynamic Vehicle Interference** | White SUV and passing vehicles cross ego-lane corridor | Lanes stay mapped strictly to asphalt markings | OpenCV Hough lines snapped to car bumper edges due to high contrast | Identified as a fundamental limit of non-semantic CV; documented for deep learning segmentation upgrade. |
| **TC-04: Model Dependency Integrity** | Execution of MiDaS depth model from PyTorch Hub | Clean weight loading and depth map generation | Pipeline crashed with `ModuleNotFoundError: No module named 'timm'` | Added `timm` to virtual environment dependencies and locked it in `requirements.txt`. |