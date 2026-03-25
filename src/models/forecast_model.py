import torch
import torch.nn as nn


class LSTMForecast(nn.Module):
    """
    LSTM-based forecasting model with dropout
    (supports uncertainty via MC dropout)
    """

    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)

        # Take last timestep
        out = out[:, -1, :]

        out = self.dropout(out)

        return self.fc(out)