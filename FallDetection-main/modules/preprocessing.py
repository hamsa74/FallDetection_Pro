import cv2

def enhance_frame(frame):
    img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    frame = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    return frame