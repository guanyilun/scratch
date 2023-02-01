#%%
import torch.nn as nn
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self, idim, odim, hidden=[64, 64]):
        super(Net, self).__init__()
        self.layer0 = nn.Linear(idim, hidden[0])
        for i in range(1, len(hidden)):
            self.add_module('layer'+str(i), nn.Linear(hidden[i-1], hidden[i-1]))
        self.add_module('layer'+str(len(hidden)), nn.Linear(hidden[-1], odim))

    def forward(self, out):
        for i in range(len(self._modules)-1):
            out = self._modules['layer'+str(i)](out)
            out = F.relu(out)
        out = self._modules['layer'+str(len(self._modules)-1)](out)
        return out