import os
import torch
from torch.utils.data import DataLoader
from dataset import OTBMultiSequenceDataset
from rolo_model import ROLOModel

# Configuración
ROOT_DIR = 'OTB100'  # Ruta donde están las 30 carpetas
SEQ_LEN = 10
BATCH_SIZE = 1
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset y DataLoader
dataset = OTBMultiSequenceDataset(root_dir=ROOT_DIR, seq_len=SEQ_LEN, device=DEVICE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Obtener dimensión de features
sample_input, _ = dataset[0]  # (seq_len, feat_dim + 4)
input_size = sample_input.shape[1]

# Modelo, pérdida y optimizador
model = ROLOModel(input_size=input_size, hidden_size=256, num_layers=1).to(DEVICE)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Entrenamiento
model.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for i, (seq_inputs, target_box) in enumerate(dataloader):
        seq_inputs = seq_inputs.to(DEVICE)        # (B, T, D)
        target_box = target_box.to(DEVICE)        # (B, 4)

        optimizer.zero_grad()
        pred_box = model(seq_inputs)              # (B, 4)
        loss = criterion(pred_box, target_box)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {avg_loss:.4f}")

# Guardar modelo
os.makedirs('checkpoints', exist_ok=True)
torch.save(model.state_dict(), 'checkpoints/rolo_lstm.pth')
print("Modelo guardado en checkpoints/rolo_lstm.pth")