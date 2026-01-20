import torch
import torch.nn as nn

class DC_PF(nn.Module):
    """ Implements the DC Power Flow as a neural network module."""
    def __init__(self):
        super().__init__()

    def use_physics_loss(self):
        return False

    def is_supervised(self):
        return False

    def is_complex(self):
        return False

    def is_analytical(self):
        return True

    def forward(self, data):
        return data.dc_pf

class DC_PF_Slack(DC_PF):
    """ Sets all voltage magnitudes to slack bus voltage magnitude."""
    def __init__(self):
        super().__init__()

    def forward(self, data):
        out = data.dc_pf
        out[:, 0] = torch.ones(len(out))*data.slack_info[0]
        return out