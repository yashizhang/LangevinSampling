import numpy as np
import torch
from montecarlo import MonteCarlo

class LangevinMonteCarlo(MonteCarlo):
    def __init__(self, potential_func, step_size, num_steps, device='cpu'):
        super().__init__(potential_func, num_steps, device)
        self.step_size = step_size

    def sample(self, initial_state):
        current_state = initial_state.clone().to(self.device)
        samples = []

        for _ in range(self.num_steps):
            current_state = self._langevin_step(current_state)
            samples.append(current_state.cpu().detach().numpy())

        return np.array(samples)

    def _langevin_step(self, state):
        noise = torch.randn_like(state).to(self.device)
        grad = self._compute_gradient(state)
        state = state - self.step_size * grad + (2 * self.step_size)**0.5 * noise
        return state

    def _compute_gradient(self, state):
        state = state.clone().requires_grad_(True)
        potential = self.potential_func(state)
        grad = torch.autograd.grad(potential.sum(), state)[0]
        return grad
