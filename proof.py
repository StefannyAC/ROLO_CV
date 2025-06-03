from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import glob
import cv2
from YOLOv8FeatureExtractor import YOLOv8FeatureExtractor
import matplotlib.pyplot as plt



if __name__ == "__main__":
    model = YOLO('yolov8s.pt')


    image_path = 'OTB100/Coke/img/0001.jpg'  # Cambia esto a tu imagen
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir a RGB

    extractor = YOLOv8FeatureExtractor(model_path='yolov8s.pt', feature_layer=8)
    bbox, feature_vector = extractor.extract(image)
    if bbox is not None and feature_vector is not None:
        print("Bounding Box:", bbox)
        print("Feature Vector:", feature_vector.shape)

    # Visualizar la imagen con la bounding box
    x,y,w,h = bbox
    x1, y1 = int((x - w / 2) * image.shape[1]), int((y - h / 2) * image.shape[0])
    x2, y2 = int((x + w / 2) * image.shape[1]), int((y + h / 2) * image.shape[0])

    image_draw = image.copy()
    cv2.rectangle(image_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)
    plt.imshow(image_draw)
    plt.axis('off')