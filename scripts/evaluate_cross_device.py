import torch
from utils import EncoderNetwork, TransformerDecoder, load_trace_csv

def evaluate(trace_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    encoder = EncoderNetwork()
    decoder = TransformerDecoder()
    encoder.load_state_dict(torch.load("models/encoder.pth", map_location=device))
    decoder.load_state_dict(torch.load("models/transformer_decoder.pth", map_location=device))
    encoder.load_state_dict(torch.load("models/encoder.pth", map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load("models/transformer_decoder.pth", map_location=device, weights_only=True))
    encoder.to(device).eval()
    decoder.to(device).eval()

    # Load and process trace
    trace = load_trace_csv(trace_path).unsqueeze(0).to(device)  # Add batch dim

    with torch.no_grad():
        latent = encoder(trace)  # [1, seq, dim]
        pred = decoder(latent)   # [1, seq, classes]
        pred_labels = pred.argmax(dim=-1).squeeze()

    print("Predicted Layer Sequence:")
    print(pred_labels.cpu().numpy())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.trace)
