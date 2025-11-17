import os
import urllib.parse
from PIL import Image
import numpy as np
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path
from paddleocr import PaddleOCR

# ------------------------------
# PaddleOCR setup (CPU)
# ------------------------------
ocr = PaddleOCR(
    lang="en",
    text_detection_model_dir="C:/Users/1003380/label-studio-ml-backend/ocr/inference/det_best",
    text_recognition_model_dir="C:/Users/1003380/label-studio-ml-backend/ocr/inference/rec_best",
    device="cpu",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

# Root folder for local files
DATA_ROOT = "C:/"

# ------------------------------
# Label Studio ML Backend
# ------------------------------
class NewModel(LabelStudioMLBase):
    def __init__(self, project_id=None, label_config=None, **kwargs):
        super().__init__(project_id=project_id, label_config=label_config, **kwargs)
        self.ocr = ocr

    def load_image(self, img_path):
        """Load image from URL or local path."""
        if img_path.startswith("http://") or img_path.startswith("https://"):
            filepath = get_image_local_path(img_path)
        else:
            if not os.path.isabs(img_path):
                filepath = os.path.join(DATA_ROOT, img_path)
            else:
                filepath = img_path
        return Image.open(filepath)

    def predict(self, tasks, **kwargs):
        results = []

        for task in tasks:
            img_path_raw = task['data'].get('ocr')
            if not img_path_raw:
                results.append({"result": []})
                continue

            # Clean Label Studio URL encoding
            img_path_raw = img_path_raw.replace('/data/local-files/?d=', '')
            img_path_raw = urllib.parse.unquote(img_path_raw)

            # Load image
            try:
                IMG = self.load_image(img_path_raw)
            except FileNotFoundError:
                results.append({"result": []})
                continue

            # Convert to 3-channel RGB if needed
            IMG_np = np.array(IMG)
            if IMG_np.ndim == 2:
                IMG_np = np.stack([IMG_np] * 3, axis=-1)
            elif IMG_np.shape[2] == 4:
                IMG_np = IMG_np[:, :, :3]

            # PaddleOCR prediction
            ocr_result = self.ocr.predict(IMG_np)

            task_results = []
            all_scores = []

            if ocr_result and len(ocr_result) > 0:
                rec_texts = ocr_result[0].get("rec_texts", [])
                rec_scores = ocr_result[0].get("rec_scores", [])
                rec_polys = ocr_result[0].get("rec_polys", [])

                for idx, (text, score, poly) in enumerate(zip(rec_texts, rec_scores, rec_polys)):
                    # Convert polygon to bounding box
                    xs = poly[:, 0]
                    ys = poly[:, 1]
                    x_min = float(np.min(xs) / IMG.width * 100)
                    y_min = float(np.min(ys) / IMG.height * 100)
                    width = float((np.max(xs) - np.min(xs)) / IMG.width * 100)
                    height = float((np.max(ys) - np.min(ys)) / IMG.height * 100)

                    # Rectangle region with recognized text
                    region = {
                        "original_width": IMG.width,
                        "original_height": IMG.height,
                        "image_rotation": 0,
                        "value": {
                            "x": x_min,
                            "y": y_min,
                            "width": width,
                            "height": height,
                            "rotation": 0,
                            "text": [text]
                        },
                        "id": f"{task['id']}_{idx}",
                        "from_name": "transcription",
                        "to_name": "ocr",
                        "type": "textarea",
                        "origin": "auto"
                    }

                    task_results.append(region)
                    all_scores.append(score)

            avg_score = float(np.mean(all_scores)) if all_scores else 0.0

            results.append({
                "result": task_results,
                "score": avg_score
            })

        return results

    @staticmethod
    def _extract_meta(task):
        """Extract region metadata."""
        meta = {}
        if task:
            meta['id'] = task['id']
            meta['from_name'] = task['from_name']
            meta['to_name'] = task['to_name']
            meta['type'] = task['type']
            meta['x'] = task['value']['x']
            meta['y'] = task['value']['y']
            meta['width'] = task['value']['width']
            meta['height'] = task['value']['height']
            meta["original_width"] = task['original_width']
            meta["original_height"] = task['original_height']
        return meta
