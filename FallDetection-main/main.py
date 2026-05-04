import os
import sys
import cv2

# --- 1. Path Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# --- 2. Module Imports ---
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    import modules.preprocessing as preprocessing
    import modules.detection as detection
    import modules.logger_utils as logger_utils
    import modules.verification as verification
    
    print(f"MediaPipe Version: {mp.__version__}")
    print("Success: All Modules Loaded!")
except Exception as e:
    print(f"Import Warning: {e}")

# --- 3. Main Run Function ---
def run_project():
    print("Starting Fall Detection System... Press 'q' to stop.")
    
    # Initialize Camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Create one single window for display
    window_name = 'Fall Detection System'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Processing steps
        processed_frame = preprocessing.enhance_frame(frame)
        detections = detection.detect_person(processed_frame)
        
        # Draw everything on the ORIGINAL frame
        for person in detections:
            bbox = person["bbox"]
            
            # Safe Fall Check
            is_fall = False
            try:
                is_fall = verification.verify_fall(person)
            except Exception:
                is_fall = False 
            
            if is_fall:
                # Alert State (Red)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 3)
                cv2.putText(frame, "!!! FALL DETECTED !!!", (bbox[0], bbox[1]-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                detection.draw_skeleton(frame, person)
                logger_utils.log_event("Fall Detected")
            else:
                # Normal State (Green)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
                detection.draw_skeleton(frame, person)

        # Final Display (Only ONE window)
        cv2.imshow(window_name, frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Cleaning up...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_project()