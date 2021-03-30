from torch import nn
from losses import SoftDiceLoss

class DiceCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = SoftDiceLoss()

    def forward(self, net_output, target):
        dice = self.dice(net_output, target)
        ce = self.ce(net_output, target)
        return ce + dice