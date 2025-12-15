import logging
import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_image(url):
    try:
        # In a real scenario, you might need to handle authentication headers
        # if the image is protected by Label Studio's auth.
        # For now, we assume the URL is accessible.
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        return None

def template_matching(full_image, bbox, threshold=0.7):
    # bbox: [x, y, width, height] (pixels)
    x, y, w, h = bbox
    
    img_h, img_w = full_image.shape[:2]
    x = max(0, int(x))
    y = max(0, int(y))
    w = min(int(w), img_w - x)
    h = min(int(h), img_h - y)
    
    if w <= 0 or h <= 0:
        return []

    # Convert to grayscale as per mock-up
    gray_full = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
    template = gray_full[y:y+h, x:x+w]
    
    # Apply template matching
    res = cv2.matchTemplate(gray_full, template, cv2.TM_CCOEFF_NORMED)
    
    # Find candidates above threshold
    loc_y, loc_x = np.where(res >= threshold)
    scores = res[loc_y, loc_x]
    
    candidates = list(zip(loc_x, loc_y, scores))
    
    # Custom NMS from mock-up
    final_detections = []
    min_distance = w // 2
    
    # Sort by score descending
    for cx, cy, score in sorted(candidates, key=lambda c: c[2], reverse=True):
        # Filter out the original bbox (approximate match)
        if abs(cx - x) < w/2 and abs(cy - y) < h/2:
            continue

        keep = True
        for fx, fy, fscore in final_detections:
            if np.sqrt((cx-fx)**2 + (cy-fy)**2) < min_distance:
                keep = False
                break
        if keep:
            final_detections.append((cx, cy, score))
            
    suggestions = []
    for fx, fy, score in final_detections:
        suggestions.append({
            'x': int(fx),
            'y': int(fy),
            'width': int(w),
            'height': int(h),
            'score': float(score)
        })
        
    return suggestions

@app.route('/suggest', methods=['POST'])
def suggest():
    data = request.json
    # Expected data format:
    # {
    #   "image_url": "...",
    #   "bbox": { "x": ..., "y": ..., "width": ..., "height": ... }, # percentages 0-100
    #   "threshold": 0.7 (optional)
    # }
    
    image_url = data.get('image_url')
    bbox_data = data.get('bbox')
    threshold = data.get('threshold', 0.7)
    
    if not image_url or not bbox_data:
        return jsonify({'error': 'Missing image_url or bbox'}), 400
        
    image = download_image(image_url)
    if image is None:
        return jsonify({'error': 'Could not download image'}), 400
        
    img_h, img_w = image.shape[:2]
    
    # Assume input is percentages (Label Studio standard)
    x = bbox_data.get('x', 0)
    y = bbox_data.get('y', 0)
    w = bbox_data.get('width', 0)
    h = bbox_data.get('height', 0)
    
    x_px = (x / 100.0) * img_w
    y_px = (y / 100.0) * img_h
    w_px = (w / 100.0) * img_w
    h_px = (h / 100.0) * img_h
    
    bbox_px = [x_px, y_px, w_px, h_px]
    
    suggestions_px = template_matching(image, bbox_px, threshold=float(threshold))
    
    # Convert back to percentages
    suggestions_pct = []
    for s in suggestions_px:
        suggestions_pct.append({
            'x': (s['x'] / img_w) * 100.0,
            'y': (s['y'] / img_h) * 100.0,
            'width': (s['width'] / img_w) * 100.0,
            'height': (s['height'] / img_h) * 100.0,
            'score': s['score']
        })
        
    return jsonify({'suggestions': suggestions_pct})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090)
