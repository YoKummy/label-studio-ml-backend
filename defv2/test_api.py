from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Store config in memory
CONFIG = {
    "model_path": "/home/jonathanyeh/yolov9/runs/train/exp/weights/best.pt",
    "conf": 0.25,
    "version": "v1",
    "labels":{
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
}

class ConfigUpdate(BaseModel):
    model_path: str = None
    conf: float = None
    version: str = None

@app.get("/config")
def get_config():
    return CONFIG

@app.post("/config")
def update_config(update: ConfigUpdate):
    for key, value in update.dict().items():
        if value is not None:
            CONFIG[key] = value
    return CONFIG

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
