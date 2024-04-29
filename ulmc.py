import numpy as np
import torch
from montecarlo import MonteCarlo

class UnderdampedLangevinMonteCarlo(MonteCarlo):
    def __init__(self, potential_func, step_size, num_steps, gamma, device='cpu'):
        super().__init__(potential_func, num_steps, device)
        self.step_size = step_size
        self.gamma = gamma

    def sample(self, initial_state, initial_velocity):
        current_state = initial_state.clone().to(self.device)
        current_velocity = initial_velocity.clone().to(self.device)
        samples = []

        for _ in range(self.num_steps):
            current_state, current_velocity = self._ulmc_step(current_state, current_velocity)
            samples.append(current_state.cpu().detach().numpy())

        return np.array(samples)

    def _ulmc_step(self, state, velocity):
        noise = torch.randn_like(velocity).to(self.device)
        grad = self._compute_gradient(state)

        # Update velocity
        velocity = velocity - self.step_size * grad - self.gamma * self.step_size * velocity + (2 * self.gamma * self.step_size)**0.5 * noise

        # Update state
        state = state + self.step_size * velocity

        return state, velocity

    def _compute_gradient(self, state):
        state = state.clone().requires_grad_(True)
        potential = self.potential_func(state)
        grad = torch.autograd.grad(potential.sum(), state)[0]
        return grad
