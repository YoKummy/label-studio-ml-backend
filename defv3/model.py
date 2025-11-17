from typing import List
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
import numpy as np
import torch
import cv2
import requests
import json

# default values (keep as comment for reference)
DEFAULT_MODEL_PATH = "/home/jonathanyeh/yolov9/runs/train/exp/weights/best.pt"  # original path
DEFAULT_CONF = 0.25  # original confidence threshold
DEFAULT_VERSION = "v1"
DEFAULT_LABELS = {  # original label dict
    0: 'Printing Defect',
    1: 'Hole Bias',
    2: 'Foreign Object',
    3: 'Reflector Oil Stain',
    4: 'Reflector Bubbles',
    5: 'Reflector Crease',
    6: 'Reflector Scratch',
    7: 'Reflector Damage',
    8: 'LGP Scratch',
    9: 'LGP Oil Stain',
    10: 'Mask Crease',
    11: 'Mask Scratch',
    12: 'Mask Glue Foreign Object',
    13: 'Mask Damage',
    14: 'Mask Oil Stain',
    15: 'Mylar Bias',
    16: 'Lack Glue',
    17: 'Glue Edge Curl',
    18: 'Edge Sealing Warping',
    19: 'Mask Glue Residue'
}

CONFIG_SERVER_URL = "http://localhost:8000/config"  # replace with your remote config server

def load_config_from_api():
    """Fetch the latest config from remote API, fallback to defaults if error"""
    try:
        r = requests.get(CONFIG_SERVER_URL, timeout=2)
        r.raise_for_status()
        config = r.json()
    except Exception as e:
        print(f"Error fetching remote config: {e}")
        config = {}
    return {
        "model_path": config.get("model_path", DEFAULT_MODEL_PATH),
        "conf": config.get("conf", DEFAULT_CONF),
        "version": config.get("version", DEFAULT_VERSION),
        "labels": config.get("labels", DEFAULT_LABELS)
    }

def load_model(path, conf):
    """Load YOLO model with given path and confidence"""
    model = torch.hub.load(
        '/home/jonathanyeh/yolov9/', 'custom',
        path=path, source='local'
    )
    model.conf = conf
    # dummy run to initialize
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    with torch.no_grad():
        model(dummy)
    return model

class NewModel(LabelStudioMLBase):
    """ML backend with remote hot-reload config via API"""
    
    def __init__(self, project_id=None, label_config=None, **kwargs):
        super().__init__(project_id=project_id, label_config=label_config, **kwargs)
        self.model = None
        self.current_model_path = None
        self.current_conf = None
        self.labels = DEFAULT_LABELS.copy()

    def setup(self):
        self.set("model_version", DEFAULT_VERSION)

    def predict(self, tasks, **kwargs) -> ModelResponse:
        predictions = []

        # Fetch remote config every predict call
        config = load_config_from_api()
        model_path = config["model_path"]
        conf = config["conf"]
        version = config["version"]
        self.labels = config["labels"]

        # Hot-reload model if path/conf changed
        if self.model is None or self.current_model_path != model_path or self.current_conf != conf:
            print(f"Hot-reloading model: {model_path} with conf {conf}")
            self.model = load_model(model_path, conf)
            self.current_model_path = model_path
            self.current_conf = conf

        # Update model version dynamically
        self.set("model_version", version)

        for task in tasks:
            img_url = task['data']['image']
            img_path = self.get_local_path(img_url, task_id=task['id'])
            img = cv2.imread(img_path)
            if img is None:
                print("No img found at:", img_path)
                continue

            with torch.no_grad():
                results = self.model(img)
                boxes = results.xyxy[0].cpu().numpy()

            detections = []
            scores = []

            for box in boxes:
                x1, y1, x2, y2, score, cls_id = box
                label = self.labels.get(int(cls_id), "unknown")
                detection = {
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": float(x1) / img.shape[1] * 100,
                        "y": float(y1) / img.shape[0] * 100,
                        "width": float(x2 - x1) / img.shape[1] * 100,
                        "height": float(y2 - y1) / img.shape[0] * 100,
                        "rectanglelabels": [label]
                    },
                    "score": float(score)
                }
                detections.append(detection)
                scores.append(float(score))

            task_score = float(min(scores)) if scores else 0.0
            predictions.append({
                "model_version": self.get("model_version"),
                "score": task_score,
                "result": detections
            })

        return ModelResponse(predictions=predictions)
