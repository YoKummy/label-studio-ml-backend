import logging
import cv2
import numpy as np
import requests
import os
import urllib.parse
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
#  CONFIGURATION
# =============================================================================
# Set this to the absolute path of your image folder.
#
# CASE 1: If you used "Import" -> "Upload Files" in Label Studio:
# Images are usually here: C:\Users\<YOU>\AppData\Local\label-studio\media\upload
#
# CASE 2: If you used "Cloud/Local Storage" -> "Local files":
# Set this to the folder you pointed Label Studio to.
# =============================================================================
# Trying to guess the default upload location based on your username
LOCAL_IMAGE_ROOT = r"C:\Users\1003380"  #C:\Users\1003380\AppData\Local\label-studio\media\upload


def get_image(url):
    """
    Tries to load image from local disk first (if configured),
    otherwise falls back to HTTP download.
    """
    # 1. Try Local Disk Access
    if LOCAL_IMAGE_ROOT and os.path.exists(LOCAL_IMAGE_ROOT):
        try:
            parsed_url = urllib.parse.urlparse(url)
            
            # Logic for "Local Storage" source (URLs look like ?d=folder/img.jpg)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            if 'd' in query_params:
                # relative_path = query_params['d'][0]
                # local_path = os.path.join(LOCAL_IMAGE_ROOT, relative_path)
                relative_path = urllib.parse.unquote(query_params['d'][0])
                relative_path = relative_path.replace('\\', os.sep).replace('/', os.sep)

                local_path = os.path.join(LOCAL_IMAGE_ROOT, relative_path)
                if os.path.exists(local_path):
                    logger.info(f"Loading local file (via ?d=): {local_path}")
                    return cv2.imread(local_path)

            # Logic for "Uploaded" files (URLs look like /data/upload/1/img.jpg)
            # We try to match the path structure
            path_parts = parsed_url.path.strip('/').split('/')
            
            # If URL is /data/upload/12/img.jpg, we want to join LOCAL_IMAGE_ROOT + 12/img.jpg
            # We iterate to find a matching subpath
            if 'upload' in path_parts:
                idx = path_parts.index('upload')
                # Join everything after 'upload'
                sub_path = os.path.join(*path_parts[idx+1:])
                local_path = os.path.join(LOCAL_IMAGE_ROOT, sub_path)
                if os.path.exists(local_path):
                    logger.info(f"Loading local file (upload match): {local_path}")
                    return cv2.imread(local_path)

        except Exception as e:
            logger.warning(f"Failed to load from local disk: {e}")

    # 2. Fallback to HTTP Download
    try:
        logger.info(f"Downloading image from URL: {url}")
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


@app.route("/health", methods=["GET"])
def health():
    return "ok"

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
        
    # Use the new smart get_image function
    image = get_image(image_url)
    
    if image is None:
        return jsonify({'error': 'Could not load image'}), 400
        
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
    print(f"Starting Zero-Shot Assist on port 9090...")
    print(f"Local Image Root: {LOCAL_IMAGE_ROOT}")
    app.run(host='0.0.0.0', port=9090)
