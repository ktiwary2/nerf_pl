import torch
from torch import nn

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return loss
               
class SMMSELoss(nn.Module):
    def __init__(self):
        super(SMMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['sm_coarse'], targets)
        if 'sm_fine' in inputs:
            loss += self.loss(inputs['sm_fine'], targets)

        return loss
loss_dict = {'mse': MSELoss, 'sm': SMMSELoss}