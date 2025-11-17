from label_studio_ml.model import LabelStudioMLBase
import torch
import cv2

class NewModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = torch.hub.load('WongKinYiu/yolov9', 'custom', path='best.pt', source='local')  # Load local model
        self.model.conf = 0.25  # confidence threshold

        self.labels = {
            0: "cat",
            1: "dog"
        }

    def predict(self, tasks, **kwargs):
        results = []
        for task in tasks:
            img_path = task['data']['image'].replace('/data/local-files/?d=', '')  # fix for local files
            img = cv2.imread(img_path)
            if img is None:
                results.append({"result": []})
                continue

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

            results.append({
                "result": detections
            })

        return results
