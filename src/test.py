import numpy as np
from torch import nn
from common_utils.losses import get_loss
ce = get_loss('ce')
dice = get_loss('dice')

print(ce._name)
print(dice._name)