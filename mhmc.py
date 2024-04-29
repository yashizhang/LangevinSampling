import numpy as np 
import torch 
from metropolis import MetropolisAdjusted

class MetropolizedHamiltonianMonteCarlo(MetropolisAdjusted):
    def __init__(self, potential_func, step_size, num_steps, leapfrog_steps, device='cpu'):
        proposal_func = lambda x: self._hmc_proposal(x, step_size, leapfrog_steps)
        super().__init__(potential_func, proposal_func, num_steps, device)
        self.step_size = step_size
        self.leapfrog_steps = leapfrog_steps

    def _hmc_proposal(self, state, step_size, leapfrog_steps):
        momentum = torch.randn_like(state).to(self.device)
        state, momentum = self._hmc_step(state, momentum, step_size, leapfrog_steps)
        return state

    def _hmc_step(self, state, momentum, step_size, leapfrog_steps):
        state = state.clone().requires_grad_(True)
        potential = self.potential_func(state)
        grad = torch.autograd.grad(potential.sum(), state)[0]

        for _ in range(leapfrog_steps):
            momentum -= 0.5 * step_size * grad
            state = state.detach() + step_size * momentum
            state.requires_grad_(True)
            potential = self.potential_func(state)
            grad = torch.autograd.grad(potential.sum(), state)[0]
            momentum -= 0.5 * step_size * grad

        return state, momentum