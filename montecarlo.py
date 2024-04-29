import numpy as np
import torch

class MonteCarlo:
    def __init__(self, potential_func, num_steps, device='cpu'):
        self.potential_func = potential_func
        self.num_steps = num_steps
        self.device = device

    def sample(self, initial_state):
        raise NotImplementedError("Subclasses should implement this method.")

