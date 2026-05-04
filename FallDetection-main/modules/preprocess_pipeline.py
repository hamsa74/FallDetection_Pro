# import numpy as np
# import os
# import random
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import cv2

# def extract_frames(video_path, num_frames=30, img_size=(224, 224)):
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     if total_frames == 0:
#         cap.release()
#         return None
    
#     indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
#     frames = []
    
#     for idx in indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         ret, frame = cap.read()
#         if ret:
#             frame = cv2.resize(frame, img_size)
#             frames.append(frame)
#         else:
#             frames.append(np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8))
    
#     cap.release()
#     return np.array(frames)

# def normalize_frames(frames):
#     return frames.astype(np.float32) / 255.0

# def augment_frame(frame):
#     if random.random() > 0.5:
#         frame = cv2.flip(frame, 1)
    
#     if random.random() > 0.5:
#         alpha = random.uniform(0.8, 1.2)
#         frame = np.clip(frame * alpha, 0, 255).astype(np.uint8)
    
#     if random.random() > 0.5:
#         angle = random.uniform(-15, 15)
#         h, w = frame.shape[:2]
#         M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
#         frame = cv2.warpAffine(frame, M, (w, h))
    
#     return frame

# def augment_video(frames):
#     augmented = []
#     for frame in frames:
#         aug_frame = augment_frame(frame)
#         augmented.append(aug_frame)
#     return np.array(augmented)

# def load_dataset(data_dir, categories=['fall', 'normal'], num_frames=30, img_size=(224, 224), augment=False):
#     X = []
#     y = []
    
#     for label, category in enumerate(categories):
#         category_path = os.path.join(data_dir, category)
#         if not os.path.exists(category_path):
#             print(f"Warning: {category_path} not found")
#             continue
        
#         video_files = [f for f in os.listdir(category_path) 
#                        if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
#         for video_file in video_files:
#             video_path = os.path.join(category_path, video_file)
#             frames = extract_frames(video_path, num_frames, img_size)
            
#             if frames is not None and len(frames) == num_frames:
#                 X.append(frames)
#                 y.append(label)
                
#                 if augment and label == 0:
#                     augmented_frames = augment_video(frames)
#                     X.append(augmented_frames)
#                     y.append(label)
    
#     return np.array(X), np.array(y)

# def prepare_data(X, y, test_size=0.2, val_size=0.1, shuffle=True):
#     X_temp, X_test, y_temp, y_test = train_test_split(
#         X, y, test_size=test_size, stratify=y, random_state=42, shuffle=shuffle
#     )
    
#     val_relative = val_size / (1 - test_size)
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_temp, y_temp, test_size=val_relative, stratify=y_temp, random_state=42, shuffle=shuffle
#     )
    
#     return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# def format_sequences(X, y, num_frames, img_height, img_width, channels=3):
#     X_reshaped = X.reshape(-1, num_frames, img_height, img_width, channels)
#     return X_reshaped, y

# def preprocess_pipeline(data_dir, num_frames=30, img_size=(224, 224), augment=False):
#     print("Loading dataset...")
#     X, y = load_dataset(data_dir, num_frames=num_frames, img_size=img_size, augment=augment)
    
#     print(f"Loaded {len(X)} samples")
#     print(f"Class distribution: {np.bincount(y)}")
    
#     print("Normalizing frames...")
#     X_normalized = np.array([normalize_frames(video) for video in X])
    
#     print("Splitting data...")
#     (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data(X_normalized, y)
    
#     print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
#     return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# if __name__ == "__main__":
#     print("Preprocessing module loaded successfully.")
    
#     # حطي مسار الفولدر اللي فيه فيديوهات الـ fall والـ normal
#     data_path = "./data" 
    
#     if os.path.exists(data_path):
#         train, val, test = preprocess_pipeline(data_path, augment=True)
#         print("Done! Data is ready for training.")
#     else:
#         print(f"Path not found: {data_path}. Please check your folders.")

import numpy as np
import os
import random
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


BLUR_KERNEL_SIZE = (5, 5)  
USE_ENHANCEMENT = True      
# ----------------------------------

def apply_image_enhancement(frame):
  
    if not USE_ENHANCEMENT:
        return frame

    
    frame = cv2.GaussianBlur(frame, BLUR_KERNEL_SIZE, 0)
    
    
    img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    frame = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    return frame

def extract_frames(video_path, num_frames=30, img_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            
            frame = cv2.resize(frame, img_size)
           
            frame = apply_image_enhancement(frame)
            frames.append(frame)
        else:
            
            frames.append(np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8))
    
    cap.release()
    return np.array(frames)

def normalize_frames(frames):
    
    return frames.astype(np.float32) / 255.0

def augment_frame(frame):
    
    
    if random.random() > 0.5:
        frame = cv2.flip(frame, 1)
    
    
    if random.random() > 0.5:
        alpha = random.uniform(0.8, 1.2)
        frame = np.clip(frame * alpha, 0, 255).astype(np.uint8)
    
    
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        frame = cv2.warpAffine(frame, M, (w, h))
    
    return frame

def augment_video(frames):
    augmented = []
    for frame in frames:
        aug_frame = augment_frame(frame)
        augmented.append(aug_frame)
    return np.array(augmented)

def load_dataset(data_dir, categories=['fall', 'normal'], num_frames=30, img_size=(224, 224), augment=False):
    X = []
    y = []
    
    for label, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        if not os.path.exists(category_path):
            print(f"Warning: Folder not found: {category_path}")
            continue
        
        video_files = [f for f in os.listdir(category_path) 
                       if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        for video_file in video_files:
            video_path = os.path.join(category_path, video_file)
            frames = extract_frames(video_path, num_frames, img_size)
            
            if frames is not None and len(frames) == num_frames:
                X.append(frames)
                y.append(label)
                
                
                if augment and label == 0:
                    augmented_frames = augment_video(frames)
                    X.append(augmented_frames)
                    y.append(label)
    
    return np.array(X), np.array(y)

def prepare_data(X, y, test_size=0.2, val_size=0.1, shuffle=True):
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42, shuffle=shuffle
    )
    
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, stratify=y_temp, random_state=42, shuffle=shuffle
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def preprocess_pipeline(data_dir, num_frames=30, img_size=(224, 224), augment=False):
    print("--- Starting Enhanced Preprocessing Pipeline ---")
    X, y = load_dataset(data_dir, num_frames=num_frames, img_size=img_size, augment=augment)
    
    if len(X) == 0:
        print("Error: No valid data samples found in directory.")
        return None
        
    print(f"Total samples loaded: {len(X)}")
    print(f"Class distribution [Fall, Normal]: {np.bincount(y)}")
    
    print("Applying Normalization...")
    X_normalized = np.array([normalize_frames(video) for video in X])
    
    print("Executing Data Split...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data(X_normalized, y)
    
    print(f"Pipeline finished. Train set size: {X_train.shape}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

if __name__ == "__main__":
    
    DATA_PATH = "modules/data/videos"
    
    if os.path.exists(DATA_PATH):
        
        dataset = preprocess_pipeline(DATA_PATH, augment=True)
        if dataset:
            print("\n[SUCCESS] Preprocessing is complete and data is ready for training.")
    else:
        print(f"\n[ERROR] Directory not found: {DATA_PATH}")
        print("Please verify the 'modules/data/videos' folder structure.")