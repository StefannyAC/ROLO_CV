import torch
import torch.nn as nn

class ROLO_LSTM(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=128, num_layers=1):
        """
        feature_dim: dimensión del vector de características (D)
        hidden_dim: dimensión del estado oculto de la LSTM
        num_layers: número de capas LSTM
        """
        super(ROLO_LSTM, self).__init__()

        self.input_dim = feature_dim + 4  # [features + bbox]
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc = nn.Linear(self.hidden_dim, 4)  # predice bbox: (x, y, w, h)

    def forward(self, x, hidden=None):
        """
        x: Tensor de entrada (batch_size, seq_len, input_dim)
        hidden: estado inicial de la LSTM (opcional)
        returns: predicciones (batch_size, seq_len, 4)
        """
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.fc(lstm_out)
        return output, hidden