import torch
from torch import nn, optim
from utils import get_latent_dataloader, TransformerDecoder

def train_decoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerDecoder().to(device)
    loader = get_latent_dataloader()

    criterion = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        total_loss = 0
        model.train()
        for x, y, x_len, y_len in loader:
            x, y = x.to(device), y.to(device)
            x_len = x_len.to(device)
            y_len = y_len.to(device)

            out = model(x)  # [batch, seq_len, vocab_size]
            out = out.log_softmax(2)  # CTC expects log probabilities
            out = out.permute(1, 0, 2)  # → [seq_len, batch, vocab_size]

            loss = criterion(out, y, x_len, y_len)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Loss={total_loss:.4f}")

    torch.save(model.state_dict(), "models/transformer_decoder.pth")

if __name__ == "__main__":
    train_decoder()
