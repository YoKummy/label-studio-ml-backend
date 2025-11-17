import sys
import os
import urllib.parse
# Adjust this path to point to your cloned YOLOv9 repo
YOLOV9_PATH = os.path.abspath("C:/Users/1003380/Desktop/yolov9-main")
sys.path.append(YOLOV9_PATH)

from label_studio_ml.model import LabelStudioMLBase

import numpy as np
import torch
import cv2

model = torch.hub.load(
            'C:/Users/1003380/Desktop/yolov9-main', 'custom', 
            path='C:/Users/1003380/Desktop/yolov9-main/catvdog.pt', source='local'
            )
model.conf = 0.25  # confidence threshold
model.to('cuda')

dummy = np.zeros((640, 640, 3), dtype=np.uint8)
with torch.no_grad():
    model(dummy)




# Set this to your images folder root (where your catvdog images are stored)
IMAGES_ROOT = r"C:\Users\1003380\Desktop\yolov9-main\catvdog\images"

class NewModel(LabelStudioMLBase):
    def __init__(self, project_id=None, label_config=None, **kwargs):
            super().__init__(project_id=project_id, label_config=label_config, **kwargs)
            self.model = model
            

            self.labels = {
            0: "cat",
            1: "dog"
            }

    def predict(self, tasks, **kwargs):
        results = []
        
        for task in tasks:
            # Get raw image path from Label Studio task, decode URL encoding
            img_path_raw = task['data']['image'].replace('/data/local-files/?d=', '')
            img_path_raw = urllib.parse.unquote(img_path_raw)

            # Extract relative path inside images folder (e.g. train/102.jpg)
            # We assume img_path_raw contains something like 'catvdog/images/train/102.jpg' or just 'train/102.jpg'
            # Adjust this according to your actual Label Studio setup.
            # If unsure, just use the basename (filename only):
            filename = os.path.basename(img_path_raw)

            # Compose the full absolute image path by joining IMAGES_ROOT + relative path
            img_path = os.path.join(IMAGES_ROOT, filename)

            if not os.path.exists(img_path):
                print(f"[ERROR] Image file not found: {img_path}")
                results.append({"result": []})
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"[ERROR] Failed to read image: {img_path}")
                results.append({"result": []})
                continue
            
            with torch.no_grad():
                predictions = self.model(img)
                boxes = predictions.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2, conf, cls)
            
            detections = []
            for box in boxes:
                x1, y1, x2, y2, conf, cls_id = box
                label = self.labels.get(int(cls_id), "unknown")

                detection = {
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": x1 / img.shape[1] * 100,
                        "y": y1 / img.shape[0] * 100,
                        "width": (x2 - x1) / img.shape[1] * 100,
                        "height": (y2 - y1) / img.shape[0] * 100,
                        "rectanglelabels": [label]
                    },
                    "score": float(conf)
                }
                detections.append(detection)

            # results.append({"result": detections})
            avg_score = float(np.mean([d['score'] for d in detections])) if detections else 0.0
            results.append({
                "result": detections,
                "score": avg_score
        })


        return results
