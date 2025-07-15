# scripts/utils.py
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

def load_trace_csv(path):
    import pandas as pd
    import torch.nn.functional as F

    df = pd.read_csv(path)
    x = torch.tensor(df.values, dtype=torch.float32)

    # Project or pad to 128 features
    if x.shape[1] < 128:
        x = F.pad(x, (0, 128 - x.shape[1]))  # Pad right side with zeros

    return x


def get_dataloaders():
    # Placeholder for contrastive trace pairs
    dummy_x = torch.randn(100, 128)
    dummy_y = torch.randn(100, 128)
    labels = torch.randint(0, 2, (100,))
    dataset = TensorDataset(dummy_x, dummy_y, labels)
    return DataLoader(dataset, batch_size=16), None

def get_latent_dataloader():
    # Dummy transformer input
    x = torch.randn(64, 32, 512)
    y = torch.randint(0, 10, (64, 16))
    x_len = torch.full((64,), 32, dtype=torch.int32)
    y_len = torch.full((64,), 16, dtype=torch.int32)
    dataset = TensorDataset(x, y, x_len, y_len)
    return DataLoader(dataset, batch_size=8)

class EncoderNetwork(nn.Module):
    def __init__(self, input_size=4):
        super(EncoderNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128)  # Output = 128 dim latent
        )

    def forward(self, x):
        return self.net(x)


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=128, num_layers=4, num_heads=8, num_classes=10):
        super(TransformerDecoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):  # x: [batch, seq, 128]
        x = self.transformer(x)
        x = self.head(x)
        return x


   