from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
import numpy as np
import torch
import cv2
import sys
import os
from dotenv import load_dotenv
import urllib.parse
load_dotenv()
YOLOV9_PATH = os.path.abspath("/home/jonathanyeh/yolov9/")
sys.path.append(YOLOV9_PATH)

model = torch.hub.load(
            '/home/jonathanyeh/yolov9/', 'custom', 
            path='/home/jonathanyeh/yolov9/runs/train/exp/weights/best.pt', source='local'
            )
model.conf = 0.25
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
with torch.no_grad():
    model(dummy)
class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    def __init__(self, project_id=None, label_config=None, **kwargs):
            super().__init__(project_id=project_id, label_config=label_config, **kwargs)
            self.model = model
            
            self.labels = {
            0: "Printing Defect",
            1: "Hole Bias",
            2: "Foreign Object",
            3: "Reflector Oil Stain",
            4: "Reflector Bubbles",
            5: "Reflector Crease",
            6: "Reflector Scratch",
            7: "Reflector Damage",
            8: "LGP Scratch",
            9: "LGP Oil Stain",
            10: "Mask Crease",
            11: "Mask Scratch",
            12: "Mask Glue Foreign Object",
            13: "Mask Damage",
            14: "Mask Oil Stain",
            15: "Mylar Bias",
            16: "Lack Glue",
            17: "Glue Edge Curl",
            18: "Edge Sealing Warping",
            19: "Mask Glue Residue"
            }

    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "v1")

    def predict(self, tasks, **kwargs) -> ModelResponse:
        predictions = []

        for task in tasks:
            img_url = task['data']['image']
            #img_path = self.get_local_path(img_url, task_id=task['id'], project_dir = "/")
            
            parsed = urllib.parse.urlparse(img_url)
            query = urllib.parse.parse_qs(parsed.query)
            if 'd' in query:
              path = query['d'][0]
              if not path.startswith('/'):
                path = '/' + path  # add leading slash
            
            # rebuild URL
                fixed_url = f"/data/local-files/?d={urllib.parse.quote(path)}"
            else:
              fixed_url = img_url

            img_path = self.get_local_path(fixed_url, task_id=task['id'], project_dir="/")
            #import urllib.parse
            #import os

            # Decode and resolve local path
            #decoded_path = urllib.parse.unquote(img_url)
            #if decoded_path.startswith('/data/local-files/?d='):
            #    decoded_path = decoded_path.replace('/data/local-files/?d=', '')
            
            #base_dir = "C:/"
            #img_path = os.path.join(base_dir, decoded_path)



            img = cv2.imread(img_path)
            if(img is None):
                print("No img")
                continue
             
            with torch.no_grad():
                results = self.model(img)
                boxes = results.xyxy[0].cpu().numpy()

            detections = []
            scores = []

            for box in boxes:
                x1, y1, x2, y2, conf, cls_id = box
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
                    "score": float(conf)
                }
                detections.append(detection)
                scores.append(float(conf))
                
            task_score = float(min(scores)) if scores else 0.0

            predictions.append({
                "model_version": self.get("model_version"),
                "score": task_score,
                "result": detections
            })

        print(f'''\
        Run prediction on {tasks}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}''')

        return ModelResponse(predictions=predictions)
"""     
    def fit(self, event, data, **kwargs):
        
        #This method is called each time an annotation is created or updated
        #You can run your logic here to update the model and persist it to the cache
        #It is not recommended to perform long-running operations here, as it will block the main thread
        #Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        #:param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        #:param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')

        


 """

