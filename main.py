import cv2
import os
from modules.detection_logic import PersonTracker
from modules.verification_logic import evaluate_posture
from modules.logger_utils import log_event

def start_engine():
    view = cv2.VideoCapture(0)
    
    if not view.isOpened():
        print("Error: Camera not accessible.")
        return

    tracker = PersonTracker()
    print("AI System is Active... Press 'q' to stop.")

    while True:
        success, img = view.read()
        if not success:
            break

        try:
            box = tracker.get_body_frame(img)
            
            if box is not None:
                is_fall = evaluate_posture(box)
                x, y, w, h = box
                
                if is_fall:
                    color = (0, 0, 255)
                    thickness = 3
                    cv2.putText(img, "WARNING: FALL DETECTED", (20, 50), 
                                cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                    log_event("Fall alert triggered.")
                else:
                    color = (0, 255, 0)
                    thickness = 2
                
                cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        
        except Exception as e:
            print(f"Error: {e}") 
            continue

        cv2.imshow('Fall Detection Pro - PUA AI Project', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    view.release()
    cv2.destroyAllWindows()
    print("System shut down safely.")

if __name__ == "__main__":
    start_engine()