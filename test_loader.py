import torch
from torch.utils.data import DataLoader
from Yolov8FeatureExtractor import YOLOv8FeatureExtractor
from MultiLoaderData import OTBMultiSequenceDataset

# Inicializar extractor
extractor = YOLOv8FeatureExtractor(model_path='yolov8s.pt', device='cuda')

# Inicializar dataset
dataset = OTBMultiSequenceDataset(root_dir='OTB100', extractor=extractor, seq_len=5)

# Usar DataLoader si quieres probar batching
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Obtener un batch
for i, (inputs, target_box) in enumerate(loader):
    print(f"Secuencia {i}")
    print(f"Inputs shape: {inputs.shape}")        # [batch, seq_len, feature_dim]
    print(f"Target bbox: {target_box.shape}")     # [batch, 4]
    print(f"First input (feature+box): {inputs[0,0,:10]}")
    break  # solo 1 muestra