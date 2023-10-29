import torch
import cv2
from PIL import Image

# Model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, etc.
model = torch.load('best.pt')
model.eval()

# Images

im1 = Image.open('img.jpg')  # PIL image
im2 = cv2.imread('img.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)

# Inference
results = model( [im1, im2], size=640) # batch of images

# Results
print(results)


