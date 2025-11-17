import os
import urllib.parse
import logging
from PIL import Image
import numpy as np
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path
from paddleocr import PaddleOCR
from dotenv import load_dotenv

load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoid OpenMP error

logger = logging.getLogger(__name__)
DATA_ROOT = "C:/"

# Initialize PaddleOCR (detection + recognition)
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
            if img_path_raw.startswith('/data/local-files/?d='):
                img_path_raw = img_path_raw.replace('/data/local-files/?d=', '')
                img_path_raw = urllib.parse.unquote(img_path_raw)

            # Load image
            try:
                IMG = self.load_image(img_path_raw)
            except FileNotFoundError:
                results.append({"result": []})
                continue

            if IMG is None:
                results.append({"result": []})
                continue

            # Convert image to 3-channel RGB for PaddleOCR
            IMG_np = np.array(IMG)
            if IMG_np.ndim == 2:
                IMG_np = np.stack([IMG_np]*3, axis=-1)
            elif IMG_np.shape[2] == 4:
                IMG_np = IMG_np[:, :, :3]

            # Run PaddleOCR (auto detection + recognition)
            ocr_result = self.ocr.predict(IMG_np)

            task_results = []
            all_scores = []

            # Iterate dict output from PaddleOCR 3.x
            for line in ocr_result:
                rec_texts = line.get("rec_texts", [])
                rec_scores = line.get("rec_scores", [])
                rec_polys = line.get("rec_polys", [])

                for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
                    all_scores.append(score)
                    xs = poly[:, 0]
                    ys = poly[:, 1]
                    x_min = float(np.min(xs) / IMG.width * 100)
                    y_min = float(np.min(ys) / IMG.height * 100)
                    width = float((np.max(xs) - np.min(xs)) / IMG.width * 100)
                    height = float((np.max(ys) - np.min(ys)) / IMG.height * 100)

                    region = {
                        "from_name": "transcription",
                        "to_name": "image",
                        "type": "textarea",
                        "value": {
                            "x": x_min,
                            "y": y_min,
                            "width": width,
                            "height": height,
                            "text": [text],
                            "rotation": 0
                        },
                        "id": f"{task['id']}_{len(task_results)}",
                        #"origin": "auto"
                    }
                    task_results.append(region)

            avg_score = float(np.mean(all_scores)) if all_scores else 0.0
            results.append({"result": task_results, "score": avg_score})

        return results
