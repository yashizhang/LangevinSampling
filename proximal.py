import numpy as np 
import torch 
from montecarlo import MonteCarlo

class ProximalSampler(MonteCarlo):
    def __init__(self, potential_func, step_size, num_steps, device='cpu'):
        super().__init__(potential_func, num_steps, device)
        self.step_size = step_size

    def sample(self, initial_state):
        current_state = initial_state.clone().to(self.device)
        samples = []

        for _ in range(self.num_steps):
            proposed_state = self._rgo_sampler(current_state)
            current_state = proposed_state
            samples.append(current_state.cpu().detach().numpy())

        return np.array(samples)

    def _rgo_sampler(self, state):
        while True:
            proposal = torch.randn_like(state).to(self.device) * (2 * self.step_size) ** 0.5 + state
            acceptance_prob = torch.exp(-self.potential_func(proposal)) / torch.exp(-self.potential_func(state))
            if torch.rand(1).item() < acceptance_prob.item():
                return proposal