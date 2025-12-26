import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioAdapter(nn.Module):
    def __init__(self):
        super(AudioAdapter, self).__init__()
        # Input: 512 (Standard OpenL3)
        # Output: 128 (Compressed & Specialized)
        self.layer1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(256, 128)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        # Normalize so that vector length is always 1.0 (Critical for Cosine Similarity)
        return F.normalize(x, p=2, dim=1)