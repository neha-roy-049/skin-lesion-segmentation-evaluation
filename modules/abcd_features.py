import cv2
import numpy as np
from skimage import measure

def compute_asymmetry(mask):
    h, w = mask.shape
    left = mask[:, :w//2]
    right = np.fliplr(mask[:, w//2:])
    top = mask[:h//2, :]
    bottom = np.flipud(mask[h//2:, :])

    min_w = min(left.shape[1], right.shape[1])
    min_h = min(top.shape[0], bottom.shape[0])

    horiz_diff = np.abs(left[:, :min_w].astype(float) - right[:, :min_w].astype(float)).mean()
    vert_diff = np.abs(top[:min_h, :].astype(float) - bottom[:min_h, :].astype(float)).mean()

    asymmetry = (horiz_diff + vert_diff) / 2.0
    return float(np.clip(asymmetry * 10, 0, 1))

def compute_border_irregularity(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, closed=True)
    hull = cv2.convexHull(contour)
    hull_perimeter = cv2.arcLength(hull, closed=True)

    if hull_perimeter == 0:
        return 0.0
    irregularity = (perimeter - hull_perimeter) / hull_perimeter
    return float(np.clip(irregularity, 0, 1))

def compute_color_variation(image_rgb, mask):
    lesion_pixels = image_rgb[mask == 1]
    if len(lesion_pixels) < 10:
        return 0.0

    pixels = lesion_pixels.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = min(6, len(pixels))
    _, _, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)

    color_spread = np.std(centers) / 255.0
    return float(np.clip(color_spread * 3, 0, 1))

def compute_diameter(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    diameter_px = max(w, h)
    normalized = diameter_px / 224.0
    return float(np.clip(normalized, 0, 1))

def extract_all_features(image_rgb, mask):
    return {
        'asymmetry': compute_asymmetry(mask),
        'border': compute_border_irregularity(mask),
        'color': compute_color_variation(image_rgb, mask),
        'diameter': compute_diameter(mask)
    }