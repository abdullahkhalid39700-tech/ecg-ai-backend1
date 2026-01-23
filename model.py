import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, x):
        w = torch.softmax(self.fc(x), dim=1)
        return (x * w).sum(dim=1)

class CNN_LSTM_Attn(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.attn = Attention(64)

        self.fc = nn.Sequential(
            nn.Linear(65, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, x, rr):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.attn(x)
        x = torch.cat([x, rr], dim=1)
        return self.fc(x).squeeze(1)
