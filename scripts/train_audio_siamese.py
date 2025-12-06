import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "audio_triplets.npz")
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "models", "audio_adapter.pth")


# --- 1. Define the Custom Neural Network (The Adapter) ---
class AudioAdapter(nn.Module):
    def __init__(self):
        super(AudioAdapter, self).__init__()
        # Input: 512 (OpenL3 size)
        # Hidden: 256
        # Output: 128 (Compressed, learned embedding)
        self.layer1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(256, 128)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        # L2 Normalize the output (Critical for Cosine Similarity to work)
        return torch.nn.functional.normalize(x, p=2, dim=1)


# --- 2. Define the Dataset Loader ---
class TripletDataset(Dataset):
    def __init__(self):
        data = np.load(DATA_PATH)
        self.anchors = torch.from_numpy(data['anchors'])
        self.positives = torch.from_numpy(data['positives'])
        self.negatives = torch.from_numpy(data['negatives'])

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        return self.anchors[idx], self.positives[idx], self.negatives[idx]


# --- 3. The Training Loop ---
def train():
    if not os.path.exists(DATA_PATH):
        print("Data missing. Run scripts/create_audio_triplets.py first.")
        return

    # Hyperparameters
    EPOCHS = 10
    BATCH_SIZE = 32
    MARGIN = 1.0  # This is the "Distance" margin for Triplet Loss

    # Setup
    dataset = TripletDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = AudioAdapter()

    # TRIPLE LOSS: The core of Few-Shot Learning
    # Learns to make dist(A,P) < dist(A,N)
    criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"🚀 Starting Training on {len(dataset)} triplets...")
    print(f"   Architecture: Linear(512->256) -> ReLU -> Linear(256->128)")

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for anchors, positives, negatives in dataloader:
            optimizer.zero_grad()

            # Forward pass (The same model processes all 3 inputs)
            emb_a = model(anchors)
            emb_p = model(positives)
            emb_n = model(negatives)

            # Calculate Loss
            loss = criterion(emb_a, emb_p, emb_n)

            # Backward pass (Update weights)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"   Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # Save the trained model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Success! Trained model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()