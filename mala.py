import numpy as np 
import torch
from metropolis import MetropolisAdjusted

class MetropolisAdjustedLangevinAlgorithm(MetropolisAdjusted):
    def __init__(self, potential_func, step_size, num_steps, device='cpu'):
        proposal_func = lambda x: self.proposal_function(x - step_size * self._compute_gradient(x), step_size)
        super().__init__(potential_func, proposal_func, num_steps, device)

    def proposal_function(self, x, step_size):
        return x + step_size * torch.randn_like(x)

    def _compute_gradient(self, state):
        state = state.clone().requires_grad_(True)
        potential = self.potential_func(state)
        grad = torch.autograd.grad(potential.sum(), state)[0]
        return grad