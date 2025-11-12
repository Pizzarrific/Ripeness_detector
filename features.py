import cv2
import numpy as np

def extract_color_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 100))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Get average color values
    features = [
        np.mean(h), np.std(h),
        np.mean(s), np.std(s),
        np.mean(v), np.std(v)
    ]
    return np.array(features)
