import cv2
import os

def run_system():
    # 1. Input Stage
    cap = cv2.VideoCapture(0) 
    
    if not cap.isOpened():
        print("Error: Camera not found!")
        return

    print("System Started... Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 2. Processing Stage
        cv2.putText(frame, "STATUS: MONITORING", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 3. Output Stage (Visual Output)
        cv2.imshow('Fall Detection Pro - PUA', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()