import numpy as np 
import torch 
from montecarlo import MonteCarlo

class HamiltonianMonteCarlo(MonteCarlo):
    def __init__(self, potential_func, step_size, num_steps, leapfrog_steps, device='cpu'):
        super().__init__(potential_func, num_steps, device)
        self.step_size = step_size
        self.leapfrog_steps = leapfrog_steps

    def sample(self, initial_state):
        current_state = initial_state.clone().to(self.device)
        samples = []

        for _ in range(self.num_steps):
            momentum = torch.randn_like(current_state).to(self.device)
            current_state, _ = self._hmc_step(current_state, momentum)
            samples.append(current_state.cpu().detach().numpy())

        return np.array(samples)

    def _hmc_step(self, state, momentum):
        state = state.clone().requires_grad_(True)
        potential = self.potential_func(state)
        grad = torch.autograd.grad(potential.sum(), state)[0]

        for _ in range(self.leapfrog_steps):
            momentum -= 0.5 * self.step_size * grad
            state = state.detach() + self.step_size * momentum
            state.requires_grad_(True)
            potential = self.potential_func(state)
            grad = torch.autograd.grad(potential.sum(), state)[0]
            momentum -= 0.5 * self.step_size * grad

        return state, momentum