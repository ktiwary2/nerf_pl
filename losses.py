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

class OpactiyLoss(nn.Module):
    def __init__(self, coeff=2000.0, sm_thres=0.4):
        super(OpactiyLoss, self).__init__()
        # self.loss = nn.MSELoss(reduction='mean')
        self.coeff = coeff
        self.sm_thres = sm_thres
        self.loss = nn.L1Loss(reduction='mean')

    def get_sm_pixels(self, targets):
        print("rgb shaep", targets.shape)
        targets = targets[:,0] + targets[:,1] + targets[:,2]
        targets = targets/3. # convert to black and white
        sm_idx = torch.where(targets > self.sm_thres)
        non_sm_idx = torch.where(targets <= self.sm_thres)
        return sm_idx[0], non_sm_idx[0]

    def forward(self, inputs, targets):
        """
        targets: {0,1} gt shadow pixels
        """
        sm_idx, non_sm_idx = self.get_sm_pixels(targets)
        # print("sm_idx",len(sm_idx), sm_idx)
        # print("non_sm_idx", len(non_sm_idx), non_sm_idx)
        if len(sm_idx) > 0 and len(non_sm_idx) > 0:
            opactiy = inputs['opacity_coarse']
            # print("opactiy", opactiy.shape)
            sm_opactiy = opactiy[sm_idx]
            non_sm_opactiy = opactiy[non_sm_idx]
            loss = self.coeff - self.loss(torch.mean(non_sm_opactiy), torch.mean(sm_opactiy))
            # loss = self.coeff + (torch.mean(non_sm_opactiy) - torch.mean(sm_opactiy))
            # loss = self.coeff + non_sm_opactiy.sum() - sm_opactiy.sum()
            # loss = torch.max(torch.tensor(0).to(loss.device), loss)
        else:
            return 0.

        if 'opacity_fine' in inputs:
            opactiy = inputs['opacity_fine']
            sm_opactiy = opactiy[sm_idx]
            non_sm_opactiy = opactiy[non_sm_idx]
            fine_loss = self.coeff - self.loss(torch.mean(non_sm_opactiy), torch.mean(sm_opactiy))
            # fine_loss = self.coeff + (torch.mean(non_sm_opactiy) - torch.mean(sm_opactiy))
            # fine_loss = self.coeff + non_sm_opactiy.sum() - sm_opactiy.sum()
            loss += fine_loss
            # loss += torch.max(torch.tensor(0).to(fine_loss.device), fine_loss)

        return loss 


loss_dict = {'mse': MSELoss, 'sm': SMMSELoss, 'opacity': OpactiyLoss}