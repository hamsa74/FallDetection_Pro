import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PersonTracker:
    def __init__(self):
        base_options = python.BaseOptions(
            model_asset_path='pose_landmarker_lite.task'
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO, 
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.timestamp = 0

    def get_body_frame(self, frame):
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        
        self.timestamp += 1
        result = self.detector.detect_for_video(mp_image, self.timestamp)
        
        if result.pose_landmarks:
            h, w, _ = frame.shape
            pts = result.pose_landmarks[0]  
            
            all_x = [p.x for p in pts]
            all_y = [p.y for p in pts]
            
            start_x = int(min(all_x) * w)
            end_x   = int(max(all_x) * w)
            start_y = int(min(all_y) * h)
            end_y   = int(max(all_y) * h)
            
            return [max(0, start_x), max(0, start_y), end_x - start_x, end_y - start_y]
        
        return None