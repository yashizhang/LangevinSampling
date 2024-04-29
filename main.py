import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lmc import LangevinMonteCarlo
from ulmc import UnderdampedLangevinMonteCarlo
from mrw import MetropolisRandomWalk
from mala import MetropolisAdjustedLangevinAlgorithm
from hmc import HamiltonianMonteCarlo
from mhmc import MetropolizedHamiltonianMonteCarlo

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
sk_seed = 42

def potential_function(x):
    # Example potential function: -log(exp(-0.5 * (x - 2)**2) + exp(-0.5 * (x + 2)**2))
    return -torch.log(torch.exp(-0.5 * (x - 2)**2) + torch.exp(-0.5 * (x + 2)**2))

def potential_function_numpy(x):
    return -np.log(np.exp(-0.5 * (x - 2)**2) + np.exp(-0.5 * (x + 2)**2))

def plot_samples(samples, method_name):
    plt.figure(figsize=(10, 6))
    x = np.linspace(-5, 5, 100)
    y = np.exp(-potential_function_numpy(x)) * 0.2

    plt.plot(x, y, label='Target Density')
    plt.hist(samples, bins=50, density=True, alpha=0.5, label=f'{method_name} Samples')
    plt.legend()
    plt.xlabel('Position')
    plt.ylabel('Target')
    plt.show()

if __name__ == '__main__':

    # This is basic sampling functionality
    # Set up the samplers
    num_steps = 10000
    initial_state = torch.zeros(1)  # Starting point for the samplers
    initial_velocity = torch.randn(1)  # Initial velocity for ULMC

    lmc_step_size = 0.1
    lmc_sampler = LangevinMonteCarlo(potential_function, lmc_step_size, num_steps)

    mrw_step_size = 0.1
    mrw_sampler = MetropolisRandomWalk(potential_function, mrw_step_size, num_steps)

    mala_step_size = 0.1
    mala_sampler = MetropolisAdjustedLangevinAlgorithm(potential_function, mala_step_size, num_steps)

    ulmc_step_size = 0.1
    ulmc_gamma = 1.0
    ulmc_sampler = UnderdampedLangevinMonteCarlo(potential_function, ulmc_step_size, num_steps, ulmc_gamma)

    hmc_step_size = 0.1
    hmc_leapfrog_steps = 10
    hmc_sampler = HamiltonianMonteCarlo(potential_function, hmc_step_size, num_steps, hmc_leapfrog_steps)

    mhmc_step_size = 0.1
    mhmc_leapfrog_steps = 10
    mhmc_sampler = MetropolizedHamiltonianMonteCarlo(potential_function, mhmc_step_size, num_steps, mhmc_leapfrog_steps)

    # Run the samplers
    lmc_samples = lmc_sampler.sample(initial_state)
    mrw_samples = mrw_sampler.sample(initial_state)
    mala_samples = mala_sampler.sample(initial_state)
    ulmc_samples = ulmc_sampler.sample(initial_state, initial_velocity)
    hmc_samples = hmc_sampler.sample(initial_state)
    mhmc_samples = mhmc_sampler.sample(initial_state)

    # Plot the movement of the samples in the potential function
    plot_samples(lmc_samples, 'LMC')
    plot_samples(mrw_samples, 'MRW')
    plot_samples(mala_samples, 'MALA')
    plot_samples(ulmc_samples, 'ULMC')
    plot_samples(hmc_samples, 'HMC')
    plot_samples(mhmc_samples, 'MHMC')
    

# Generate a toy dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, n_repeated=0,
                           n_classes=2, random_state=sk_seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Generate a toy regression dataset 
X = np.random.uniform(-5, 5, size=(1000, 1))
y = np.sin(X) + 0.1 * np.random.randn(1000, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
