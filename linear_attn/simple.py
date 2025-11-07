import torch.nn as nn
import torch.nn.functional as F

class SimpleBlock(nn.Module):
    def __init__(self, hidden_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim * 1.5))
        self.fc2 = nn.Linear(int(hidden_dim * 1.5), hidden_dim)

    def forward(self, hidden_states=None, **kwargs):
        x = hidden_states if hidden_states is not None else kwargs.get("x")
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x, None

# todo: videet add other linear attention variant things here