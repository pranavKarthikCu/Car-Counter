# Car-Counter
# Car Counter

A simple project to count cars in a video using computer vision.  
It uses YOLOv8 for detecting vehicles and SORT for tracking them across frames.

---

## How it works
1. YOLOv8 detects cars in each frame.  
2. SORT keeps track of each car and gives it a unique ID.  
3. The video shows bounding boxes and IDs for each car.  

---

## Requirements
- Python 3.8+
- OpenCV
- NumPy
- PyTorch
- Ultralytics (YOLOv8)
- SORT

Install dependencies with:  
```bash
pip install -r requirements.txt
