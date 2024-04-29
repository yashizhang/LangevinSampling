import numpy as np
import torch
from metropolis import MetropolisAdjusted

class MetropolisRandomWalk(MetropolisAdjusted):
    def __init__(self, potential_func, step_size, num_steps, device='cpu'):
        proposal_func = lambda x: self.proposal_function(x, step_size)
        super().__init__(potential_func, proposal_func, num_steps, device)
    def proposal_function(self, x, step_size):
        return x + step_size * torch.randn_like(x)