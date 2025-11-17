from label_studio_ml.api import init_app
from model import CatDogYOLOv9Model

app = init_app(
    model_class=CatDogYOLOv9Model,
    model_dir='.'
)
