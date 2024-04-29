import numpy as np
import torch
from montecarlo import MonteCarlo

class MetropolisAdjusted(MonteCarlo):
    def __init__(self, potential_func, proposal_func, num_steps, device='cpu'):
        super().__init__(potential_func, num_steps, device)
        self._proposal_func = proposal_func

    def sample(self, initial_state):
        current_state = initial_state.clone().to(self.device)
        samples = []

        for _ in range(self.num_steps):
            proposed_state = self._proposal_func(current_state)
            acceptance_prob = self._compute_acceptance_prob(current_state, proposed_state)
            if torch.rand(1).item() < acceptance_prob:
                current_state = proposed_state
            samples.append(current_state.cpu().detach().numpy())

        return np.array(samples)

    def _compute_acceptance_prob(self, current_state, proposed_state):
        current_potential = self.potential_func(current_state)
        proposed_potential = self.potential_func(proposed_state)
        acceptance_prob = torch.exp(current_potential - proposed_potential)
        return min(1, acceptance_prob.item())
    
    def _proposal_func(self, x):
        raise NotImplementedError("Subclasses should implement this method.")