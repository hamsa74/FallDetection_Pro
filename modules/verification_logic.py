def evaluate_posture(box_coords):
    if box_coords is None: return False
    
    x, y, width, height = box_coords
    
    posture_ratio = width / float(height)
    
    if posture_ratio > 1.2:
        return True
    return False