def verify_fall(bbox):
    if bbox:
        w, h = bbox[2], bbox[3]
        return (w / h) > 1.2
    return False