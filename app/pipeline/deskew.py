import cv2
import numpy as np

def deskew_plate(crop: np.ndarray) -> np.ndarray:
    """
    Transforms a skewed plate (trapezoid) into a perfect flat rectangle.
    Uses edge detection and cv2.warpPerspective.
    If no clear plate edges are found, returns the original crop.
    """
    if crop.size == 0:
        return crop
        
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Filter noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Edge detection
    edged = cv2.Canny(blur, 50, 200)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area, keep top 5
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    screenCnt = None
    for c in contours:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        
        # If our approximated contour has 4 points, we can assume we found the plate
        if len(approx) == 4:
            screenCnt = approx
            break
            
    if screenCnt is None:
        return crop # Fallback if no 4 points found
        
    # Order points to standard format [top-left, top-right, bottom-right, bottom-left]
    pts = screenCnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    
    # Compute width and height of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
        
    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(crop, M, (maxWidth, maxHeight))
    
    return warped
