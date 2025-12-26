import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Path Setup
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.model_def import AudioAdapter

TRIPLETS_PATH = os.path.join(ROOT_DIR, "data", "audio_triplets.npz")
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "models", "audio_adapter.pth")


class TripletDataset(Dataset):
    def __init__(self):
        if not os.path.exists(TRIPLETS_PATH):
            raise FileNotFoundError(f"Run create_audio_triplets.py first! Missing: {TRIPLETS_PATH}")
        data = np.load(TRIPLETS_PATH)
        self.anchors = torch.from_numpy(data['anchors']).float()
        self.positives = torch.from_numpy(data['positives']).float()
        self.negatives = torch.from_numpy(data['negatives']).float()

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        return self.anchors[idx], self.positives[idx], self.negatives[idx]


def train():
    print("--- Starting Training ---")

    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        dataset = TripletDataset()
    except Exception as e:
        print(e)
        return

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = AudioAdapter().to(device)

    # Triplet Loss: Pushes (A, P) together, pulls (A, N) apart
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 2. Train Loop
    model.train()
    for epoch in range(10):  # 10 Epochs is usually enough for this size
        total_loss = 0
        for anchors, positives, negatives in dataloader:
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)

            optimizer.zero_grad()
            emb_a = model(anchors)
            emb_p = model(positives)
            emb_n = model(negatives)

            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/10 | Loss: {total_loss / len(dataloader):.4f}")

    # 3. Save
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()