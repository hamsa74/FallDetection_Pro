import cv2
import mediapipe as mp

# Parameter Tuning for Robustness
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MODEL_COMPLEXITY = 1 

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose_engine = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)

def detect_person(frame):
    """
    Core detection logic using MediaPipe Pose Estimation.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_engine.process(frame_rgb)
    
    detections = []
    
    if results.pose_landmarks:
        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark
        
        # Calculate Bounding Box for Visual Output
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        
        x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
        y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
        
        padding = 20
        bbox = [
            max(0, x_min - padding), 
            max(0, y_min - padding), 
            min(w, (x_max - x_min) + 2*padding), 
            min(h, (y_max - y_min) + 2*padding)
        ]
        
        # Package data for Verification and Visual Overlays
        person_data = {
            "bbox": bbox,
            "landmarks": landmarks,
            "results": results 
        }
        
        detections.append(person_data)
        
    return detections

def draw_skeleton(frame, person_data):
    """
    Visual Output: Renders the skeletal overlay on the frame.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    if person_data["results"].pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            person_data["results"].pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )