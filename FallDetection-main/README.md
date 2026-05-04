Fall Detection with Verification
Project Overview
This project addresses the critical need for automated monitoring of elderly or recovering patients in indoor environments. Our system goes beyond simple detection by implementing a Verification Engine that distinguishes between actual falls and "fall-like" activities (e.g., sitting down quickly) using temporal and geometric analysis.
System Architecture
The project follows a complete, modular processing pipeline as required by the AI306 course:
1-Input Stage: Loading and processing pre-recorded video data.
2-Pre-processing: Background subtraction and noise reduction to isolate the subject
3-Feature Extraction: Utilizing Pose Estimation to extract human skeletal keypoints.
4-Verification Engine (The Added Value):
  -Geometric Analysis: Torso angle relative to the floor and bounding box aspect ratio.
  -Temporal Verification: State monitoring over several frames to confirm permanent falls.
5-Output Stage: Live visual alerts with status indicators ("Normal", "Potential Fall", "Fall Confirmed")
Technical Stack
-Language: Python
-Core Libraries: OpenCV & NumPy
-Pose Estimation: Mediapipe or YOLO-Pose
