import torch
from ultralytics import YOLO
import numpy as np
import torchvision.transforms as T
from PIL import Image
import cv2

class YOLOv8FeatureExtractor:
    def __init__(self, model_path='yolov8s.pt', device=None, feature_layer=-2):
        """
        model_path: Ruta al modelo YOLOv8 (p.ej. 'yolov8s.pt')
        device: 'cuda' o 'cpu'
        feature_layer: Índice de la capa desde donde extraer características
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path).to(self.device)
        self.model.eval()
        self.feature_layer = feature_layer

    def extract(self, image):
        """
        Procesa una imagen y devuelve bbox y feature vector.
        image: numpy array (HxWx3) en formato RGB
        returns: bbox (x, y, w, h) normalizado y feature vector
        """
        # === Inferencia (detección) ===
        results = self.model(image)[0]

        # === Mostrar la detección con la mayor confianza ===
        best = max(results.boxes.data, key=lambda b: b[4])  # mayor score
        x1, y1, x2, y2, conf, cls = best.tolist()
        print(f"Clase: {int(cls)}, Confianza: {conf:.2f}")
        print(f"Caja: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
        # Normalizar bbox
        x_center = (x1 + x2) / 2 / image.shape[1]
        y_center = (y1 + y2) / 2 / image.shape[0]
        w = (x2 - x1) / image.shape[1]
        h = (y2 - y1) / image.shape[0]
        bbox = np.array([x_center, y_center, w, h])

        # Preprocesar imagen
        img_resized = cv2.resize(image, (640, 640))
        tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        # === Pasar secuencialmente por bloques que aceptan entrada simple ===
        with torch.no_grad():
            x = tensor
            for i, block in enumerate(self.model.model.model):
                # Rompe antes de llegar a capa que hace cat()
                if "Concat" in block.__class__.__name__ or "Detect" in str(block):
                    print(f"Parando en bloque {i}: {block.__class__.__name__}")
                    break
                x = block(x)

            features = x
            print(f"Features shape: {features.shape}")

            vector = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1)).view(-1)
            print(f"Vector final: {vector.shape}")

            vector = vector.cpu().numpy()  # Convertir a numpy array

            #Esta es una prueba para ver si funciona el commit
        return bbox, vector, int(cls)