# **Zero-Model Annotation Assist — TODO**

A lightweight feature that auto-suggests similar objects after the annotator labels the first one.
Powered by classical CV (template matching + optional ORB), toggleable in the labeling UI.

---

## **1. Backend (CV Sidecar Service)**

* [x] Create a small service (Flask or FastAPI) running on a separate port (e.g. 9090)
* [x] Implement endpoint: `POST /suggest`
  * Input: full image + bounding box coordinates
  * Output: list of suggested bounding boxes
* [x] Implement basic template matching (OpenCV `matchTemplate`)
* [x] Add thresholding and NMS to remove duplicate matches
* [ ] (Optional) Implement ORB-based matching to reduce false positives
* [ ] Combine template + ORB for hybrid mode
* [x] Return results in Label Studio-friendly format
* [ ] Add debug mode (show heatmaps / candidate matches)

---

## **2. Label Studio Integration**

* [x] Identify frontend event triggered when user finishes drawing a bounding box
* [x] Add hook to call the sidecar service with latest annotation
* [x] Insert suggested bounding boxes into current task as pre-annotations
* [ ] Ensure suggestions are visually distinct (e.g. dotted outline)
* [x] Allow user to accept/reject suggestions (Standard LS functionality)
* [x] Make sure nothing breaks multi-object tasks

---

## **3. UI / UX**

* [x] Add a toggle: “Template Assist: ON/OFF”
  * Location: settings panel
  * State should persist per session
* [x] Add visual feedback while suggestions are loading
* [x] Add small notification when suggestions appear
* [x] Ensure suggestions don’t interrupt annotation flow

---

## **4. Configuration**

* [x] Add optional settings inside LS config:
  * [x] Matching threshold
  * [ ] Max suggestions
  * [ ] ORB on/off
* [x] Store defaults in project settings
* [ ] Add environment variable to enable globally (e.g. `LS_TM_ASSIST=true`)

---

## **5. Testing**

* [ ] Test on simple repeated objects (e.g. cats, logos, bottle caps)
* [ ] Test on cluttered backgrounds
* [ ] Test speed on large images
* [ ] Ensure fallback behavior when service fails
* [ ] Confirm compatibility with active learning later

---

## **6. Future Extensions (Not Required Now)**

* [ ] Support polygons & keypoints, not just boxes
* [ ] Auto-tune threshold per image
* [ ] Cache computed features per task
* [ ] Provide API docs for external use

---

## **Reference Prototype (Python)**

```python
"""
The following is the mock up version in python, my objective is to let user annotate something, and return the selected region to backend(flask or ml backend), and then use template matching to predict and give user similar result for auto annotation
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("muffin.jpg")
print(image.shape)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)

x, y, w, h = cv2.selectROI("Select Template", gray, showCrosshair=True, fromCenter=False)
cv2.destroyAllWindows()
template = gray[y:y+h, x:x+w]

w, h = template.shape[::-1]

result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.35

# loc = np.where(result >= threshold)
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(gray, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
# cv2.imwrite("matched.png", gray)

y, x = np.where(result >= threshold)
scores = result[y, x]

candidates = list(zip(x, y, scores))

final_detections = []
min_distance = w // 2
for x, y, score in sorted(candidates, key=lambda c: c[2], reverse=True):
    keep = True
    for fx, fy, fscore in final_detections:
        if np.sqrt((x-fx)**2 + (y-fy)**2) < min_distance:
            keep = False
            break
    if keep:
        final_detections.append((x, y, score))

# 7. Draw bounding boxes on the original image
for x, y, _ in final_detections:
    cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imwrite("detected_muffins.png", gray)

cv2.imshow("Detected Muffins", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
