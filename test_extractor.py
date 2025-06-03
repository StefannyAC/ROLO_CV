import torch
import cv2
from ultralytics import YOLO
import numpy as np

# === Cargar modelo YOLOv8 small ===
model = YOLO("yolov8s.pt")  
# === Cargar imagen ===
img_path = 'OTB100/Coke/img/0001.jpg'
img = cv2.imread(img_path)
assert img is not None, "No se pudo cargar la imagen."

# === Inferencia (detección) ===
results = model(img)[0]

# === Mostrar la detección con la mayor confianza ===
best = max(results.boxes.data, key=lambda b: b[4])  # mayor score
x1, y1, x2, y2, conf, cls = best.tolist()
print(f"Clase: {int(cls)}, Confianza: {conf:.2f}")
print(f"Caja: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")

# === Dibujar bbox sobre imagen ===
img_bbox = img.copy()
cv2.rectangle(img_bbox, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
cv2.namedWindow("Detección", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detección", 800, 600)  # Puedes ajustar el tamaño
cv2.imshow("Detección", img_bbox)
cv2.waitKey(0)
cv2.destroyAllWindows()

# === Obtener features internas (penúltima capa antes de detección) ===

# Preprocesar imagen para PyTorch
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (640, 640))
tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # shape: (1, 3, 640, 640)
tensor = tensor.to(model.device)

# === Obtener features internas sin errores con bloques secuenciales ===

# Preprocesar imagen
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (640, 640))
tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
tensor = tensor.to(model.device)

# === Pasar secuencialmente por bloques que aceptan entrada simple ===
with torch.no_grad():
    x = tensor
    for i, block in enumerate(model.model.model):
        # Rompe antes de llegar a capa que hace cat()
        if "Concat" in block.__class__.__name__ or "Detect" in str(block):
            print(f"Parando en bloque {i}: {block.__class__.__name__}")
            break
        x = block(x)

    features = x
    print(f"Features shape: {features.shape}")

    vector = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1)).view(-1)
    print(f"Vector final: {vector.shape}")