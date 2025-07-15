# scripts/train_encoder.py
import torch
from torch import nn, optim
from utils import get_dataloaders, EncoderNetwork

def train_encoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncoderNetwork().to(device)
    train_loader, _ = get_dataloaders()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):
        model.train()
        total_loss = 0
        for x1, x2, label in train_loader:
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            z1 = model(x1)
            z2 = model(x2)
            loss = criterion(z1, z2)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch}: Loss={total_loss:.4f}")

    torch.save(model.state_dict(), "models/encoder.pth")


if __name__ == "__main__":
    train_encoder()
