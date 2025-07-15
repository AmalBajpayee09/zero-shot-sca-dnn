# Zero-Shot DNN Model Extraction via Side-Channel Translation

This repository contains code and data for our paper:

> **Zero-Shot Deep Neural Network Model Extraction Across Heterogeneous Devices Using Side-Channel Translation and Foundation Models**  
> *Amal Bajpayee, Member, IEEE*

## 🔥 Highlights

- Cross-device side-channel attack for DNN extraction
- Uses contrastive encoder + transformer decoder
- Tested on Jetson Nano, RTX 3090, RX 580
- Achieves 78.6% accuracy in zero-shot cross-device recovery

## 🧠 Project Structure

| Folder | Purpose |
|--------|---------|
| `scripts/` | Train and evaluate models |
| `models/` | Pretrained encoder + decoder |
| `data/` | Side-channel trace samples |
| `notebooks/` | Demo notebook |
| `figures/` | Paper visuals |

## 🚀 Setup

```bash
git clone https://github.com/your-username/zero-shot-dnn-sca
cd zero-shot-dnn-sca
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
