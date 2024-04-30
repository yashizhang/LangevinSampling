import os
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
from models import BayesianLogisticRegression, TwoLayerNN


# What experiment to run:
# 1. Sampling from a simple potential function
# 2. Toy Bayesian logistic regression model - classification - Gaussian Prior
# 3. Toy Bayesian logistic regression model - classification - Laplace Prior
# 4. High-dimensional Bayesian logistic regression model - classification - Gaussian Prior
# 5. High-dimensional Bayesian logistic regression model - classification - Laplace Prior
# 6. Sampling from a neural network model - regression
print("Choose an experiment to run:")
print("1. Sampling from a simple potential function")
print("2. Sampling from a Bayesian logistic regression model - classification - Gaussian Prior")
print("3. Sampling from a Bayesian logistic regression model - classification - Laplace Prior")
print("4. High-dimensional Bayesian logistic regression model - classification - Gaussian Prior")
print("5. High-dimensional Bayesian logistic regression model - classification - Laplace Prior")
print("6. Sampling from a neural network model - regression (NOT IMPLEMENTED)")
option = int(input("Enter the experiment to run: "))
if option not in [1, 2, 3, 4, 5, 6]:
    raise ValueError("Invalid option. Please enter a valid option.")
if option == 6:
    raise NotImplementedError("Option 4 is not implemented yet.")

#option = 4 # DEBUGGING

# Set random seeds for reproducibility
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
sk_seed = seed

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
    plt.hist(samples, bins=50, density=True, alpha=0.5, label=f'{method_name} Empirical Density')
    plt.legend()
    plt.xlabel('Position')
    plt.ylabel('Target')
    plt.savefig(f'{method_name}.pdf', format='pdf', dpi=600)
    plt.show()
    plt.close()

if __name__ == '__main__':
    if option == 1:
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

    elif option == 2:
        def plot_contour(method_name, weights, model, y_test):
            plt.figure(figsize=(8, 6))    
            xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
            X_grid = np.c_[xx.ravel(), yy.ravel()]
            Z = model.predict(X_grid, weights).reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap='RdBu', alpha=0.3)
            plt.colorbar(label='Predicted Labels')
            plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolors='k')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title(f'Predictive Function for {method_name} and Gaussian Prior')
            plt.tight_layout()
            plt.savefig(f'blr_contour_{method_name}_l2.pdf', format='pdf', dpi=600)
            plt.close()

        # Generate a toy dataset
        X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=sk_seed)
        # Normalize the features
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=sk_seed)

        # Instantiate the samplers
        num_samples = 1000
        step_size = 0.01
        num_steps = 100
        num_features = X_train.shape[1]
        prior_weight = 10.0

        # Bayesian logistic regression with different samplers
        blr_lmc = BayesianLogisticRegression(num_features, num_samples, X_train, y_train, X_test, y_test)
        blr_mala = BayesianLogisticRegression(num_features, num_samples, X_train, y_train, X_test, y_test)
        blr_hmc = BayesianLogisticRegression(num_features, num_samples, X_train, y_train, X_test, y_test)

        blr_lmc.set_prior('gaussian', prior_weight)
        blr_mala.set_prior('gaussian', prior_weight)
        blr_hmc.set_prior('gaussian', prior_weight)

        lmc_sampler = LangevinMonteCarlo(blr_lmc.potential_func, step_size, num_steps)
        mala_sampler = MetropolisAdjustedLangevinAlgorithm(blr_mala.potential_func, step_size, num_steps)
        hmc_sampler = HamiltonianMonteCarlo(blr_hmc.potential_func, step_size, num_steps, leapfrog_steps=10)

        blr_lmc.set_sampler(lmc_sampler)
        blr_mala.set_sampler(mala_sampler)
        blr_hmc.set_sampler(hmc_sampler)

        # Sample weights
        lmc_weights = blr_lmc.sample()
        mala_weights = blr_mala.sample()
        hmc_weights = blr_hmc.sample()

        # Make predictions
        lmc_predictions = blr_lmc.predict(X_test, lmc_weights)
        mala_predictions = blr_mala.predict(X_test, mala_weights)
        hmc_predictions = blr_hmc.predict(X_test, hmc_weights)

        # Rounded predictions
        rounded_lmc_predictions = np.round(lmc_predictions).astype(int)
        rounded_mala_predictions = np.round(mala_predictions).astype(int)
        rounded_hmc_predictions = np.round(hmc_predictions).astype(int)

        # Evaluate accuracy
        lmc_accuracy = accuracy_score(y_test, rounded_lmc_predictions)
        mala_accuracy = accuracy_score(y_test, rounded_mala_predictions)
        hmc_accuracy = accuracy_score(y_test, rounded_hmc_predictions)

        # Plot the predicted contour for LMC 
        plot_contour('LMC', lmc_weights, blr_lmc, y_test)
        plot_contour('MALA', mala_weights, blr_mala, y_test)
        plot_contour('HMC', hmc_weights, blr_hmc, y_test)

        # Save the accuracies to a file
        if 'blr_l2_accuracies.txt' in os.listdir():
            os.remove('blr_l2_accuracies.txt')
        with open('blr_l2_accuracies.txt', 'w') as f:
            f.write(f'LMC: {lmc_accuracy:.4f}\n')
            f.write(f'MALA: {mala_accuracy:.4f}\n')
            f.write(f'HMC: {hmc_accuracy:.4f}\n')

    elif option == 3:
        def plot_contour(method_name, weights, model, y_test):
            plt.figure(figsize=(8, 6))    
            xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
            X_grid = np.c_[xx.ravel(), yy.ravel()]
            Z = model.predict(X_grid, weights).reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap='RdBu', alpha=0.3)
            plt.colorbar(label='Predicted Labels')
            plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolors='k')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title(f'Predictive Function for {method_name} and Laplacian Prior')
            plt.tight_layout()
            plt.savefig(f'blr_contour_{method_name}_l1.pdf', format='pdf', dpi=600)
            plt.close()
        # Generate a toy dataset
        X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=sk_seed)
        # Normalize the features
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=sk_seed)

        # Instantiate the samplers
        num_samples = 1000
        step_size = 0.01
        num_steps = 100
        num_features = X_train.shape[1]
        prior_weight = 10.0

        # Bayesian logistic regression with different samplers
        blr_lmc = BayesianLogisticRegression(num_features, num_samples, X_train, y_train, X_test, y_test)
        blr_mala = BayesianLogisticRegression(num_features, num_samples, X_train, y_train, X_test, y_test)
        blr_hmc = BayesianLogisticRegression(num_features, num_samples, X_train, y_train, X_test, y_test)

        blr_lmc.set_prior('laplace', prior_weight)
        blr_mala.set_prior('laplace', prior_weight)
        blr_hmc.set_prior('laplace', prior_weight)

        lmc_sampler = LangevinMonteCarlo(blr_lmc.potential_func, step_size, num_steps)
        mala_sampler = MetropolisAdjustedLangevinAlgorithm(blr_mala.potential_func, step_size, num_steps)
        hmc_sampler = HamiltonianMonteCarlo(blr_hmc.potential_func, step_size, num_steps, leapfrog_steps=10)

        blr_lmc.set_sampler(lmc_sampler)
        blr_mala.set_sampler(mala_sampler)
        blr_hmc.set_sampler(hmc_sampler)

        # Sample weights
        lmc_weights = blr_lmc.sample()
        mala_weights = blr_mala.sample()
        hmc_weights = blr_hmc.sample()

        # Make predictions
        lmc_predictions = blr_lmc.predict(X_test, lmc_weights)
        mala_predictions = blr_mala.predict(X_test, mala_weights)
        hmc_predictions = blr_hmc.predict(X_test, hmc_weights)

        # Rounded predictions
        rounded_lmc_predictions = np.round(lmc_predictions).astype(int)
        rounded_mala_predictions = np.round(mala_predictions).astype(int)
        rounded_hmc_predictions = np.round(hmc_predictions).astype(int)

        # Evaluate accuracy
        lmc_accuracy = accuracy_score(y_test, rounded_lmc_predictions)
        mala_accuracy = accuracy_score(y_test, rounded_mala_predictions)
        hmc_accuracy = accuracy_score(y_test, rounded_hmc_predictions)

        # Plot the predicted contour for LMC 
        plot_contour('LMC', lmc_weights, blr_lmc, y_test)
        plot_contour('MALA', mala_weights, blr_mala, y_test)
        plot_contour('HMC', hmc_weights, blr_hmc, y_test)

        # Save the accuracies to a file
        if 'blr_l1_accuracies.txt' in os.listdir():
            os.remove('blr_l1_accuracies.txt')
        with open('blr_l1_accuracies.txt', 'w') as f:
            f.write(f'LMC: {lmc_accuracy:.4f}\n')
            f.write(f'MALA: {mala_accuracy:.4f}\n')
            f.write(f'HMC: {hmc_accuracy:.4f}\n')

    elif option == 4:
        # Generate a high dimensional dataset 
        X, y = make_classification(n_samples=1000, n_features=20, n_redundant=0, n_informative=10, n_clusters_per_class=1, random_state=sk_seed)
        # Normalize the features
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=sk_seed)

        # Instantiate the samplers
        num_samples = 1000
        step_size = 0.01
        num_steps = 1000
        num_features = X_train.shape[1]
        prior_weight = 1.0

        # Bayesian logistic regression with different samplers
        blr_lmc = BayesianLogisticRegression(num_features, num_samples, X_train, y_train, X_test, y_test)
        blr_mala = BayesianLogisticRegression(num_features, num_samples, X_train, y_train, X_test, y_test)
        blr_hmc = BayesianLogisticRegression(num_features, num_samples, X_train, y_train, X_test, y_test)

        blr_lmc.set_prior('gaussian', prior_weight)
        blr_mala.set_prior('gaussian', prior_weight)
        blr_hmc.set_prior('gaussian', prior_weight)

        lmc_sampler = LangevinMonteCarlo(blr_lmc.potential_func, step_size, num_steps)
        mala_sampler = MetropolisAdjustedLangevinAlgorithm(blr_mala.potential_func, step_size, num_steps)
        hmc_sampler = HamiltonianMonteCarlo(blr_hmc.potential_func, step_size, num_steps, leapfrog_steps=10)

        blr_lmc.set_sampler(lmc_sampler)
        blr_mala.set_sampler(mala_sampler)
        blr_hmc.set_sampler(hmc_sampler)

        # Sample weights
        lmc_weights = blr_lmc.sample()
        mala_weights = blr_mala.sample()
        hmc_weights = blr_hmc.sample()

        # Make predictions
        lmc_predictions = blr_lmc.predict(X_test, lmc_weights)
        mala_predictions = blr_mala.predict(X_test, mala_weights)
        hmc_predictions = blr_hmc.predict(X_test, hmc_weights)

        # Rounded predictions
        rounded_lmc_predictions = np.round(lmc_predictions).astype(int)
        rounded_mala_predictions = np.round(mala_predictions).astype(int)
        rounded_hmc_predictions = np.round(hmc_predictions).astype(int)

        # Evaluate accuracy
        lmc_accuracy = accuracy_score(y_test, rounded_lmc_predictions)
        mala_accuracy = accuracy_score(y_test, rounded_mala_predictions)
        hmc_accuracy = accuracy_score(y_test, rounded_hmc_predictions)

        # Save the accuracies to a file
        if 'blr_l2_accuracies_high_dim.txt' in os.listdir():
            os.remove('blr_l2_accuracies_high_dim.txt')
        with open('blr_l2_accuracies_high_dim.txt', 'w') as f:
            f.write(f'LMC: {lmc_accuracy:.4f}\n')
            f.write(f'MALA: {mala_accuracy:.4f}\n')
            f.write(f'HMC: {hmc_accuracy:.4f}\n')

    elif option == 5:
        # Generate a high dimensional dataset 
        X, y = make_classification(n_samples=1000, n_features=20, n_redundant=0, n_informative=10, n_clusters_per_class=1, random_state=sk_seed)
        # Normalize the features
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=sk_seed)

        # Instantiate the samplers
        num_samples = 1000
        step_size = 0.01
        num_steps = 1000
        num_features = X_train.shape[1]
        prior_weight = 1.0

        # Bayesian logistic regression with different samplers
        blr_lmc = BayesianLogisticRegression(num_features, num_samples, X_train, y_train, X_test, y_test)
        blr_mala = BayesianLogisticRegression(num_features, num_samples, X_train, y_train, X_test, y_test)
        blr_hmc = BayesianLogisticRegression(num_features, num_samples, X_train, y_train, X_test, y_test)

        blr_lmc.set_prior('laplace', prior_weight)
        blr_mala.set_prior('laplace', prior_weight)
        blr_hmc.set_prior('laplace', prior_weight)

        lmc_sampler = LangevinMonteCarlo(blr_lmc.potential_func, step_size, num_steps)
        mala_sampler = MetropolisAdjustedLangevinAlgorithm(blr_mala.potential_func, step_size, num_steps)
        hmc_sampler = HamiltonianMonteCarlo(blr_hmc.potential_func, step_size, num_steps, leapfrog_steps=10)

        blr_lmc.set_sampler(lmc_sampler)
        blr_mala.set_sampler(mala_sampler)
        blr_hmc.set_sampler(hmc_sampler)

        # Sample weights
        lmc_weights = blr_lmc.sample()
        mala_weights = blr_mala.sample()
        hmc_weights = blr_hmc.sample()

        # Make predictions
        lmc_predictions = blr_lmc.predict(X_test, lmc_weights)
        mala_predictions = blr_mala.predict(X_test, mala_weights)
        hmc_predictions = blr_hmc.predict(X_test, hmc_weights)

        # Rounded predictions
        rounded_lmc_predictions = np.round(lmc_predictions).astype(int)
        rounded_mala_predictions = np.round(mala_predictions).astype(int)
        rounded_hmc_predictions = np.round(hmc_predictions).astype(int)

        # Evaluate accuracy
        lmc_accuracy = accuracy_score(y_test, rounded_lmc_predictions)
        mala_accuracy = accuracy_score(y_test, rounded_mala_predictions)
        hmc_accuracy = accuracy_score(y_test, rounded_hmc_predictions)

        # Save the accuracies to a file
        if 'blr_l1_accuracies_high_dim.txt' in os.listdir():
            os.remove('blr_l1_accuracies_high_dim.txt')
        with open('blr_l1_accuracies_high_dim.txt', 'w') as f:
            f.write(f'LMC: {lmc_accuracy:.4f}\n')
            f.write(f'MALA: {mala_accuracy:.4f}\n')
            f.write(f'HMC: {hmc_accuracy:.4f}\n')

    elif option == 6:
        # Generate a toy regression dataset 
        X = np.random.uniform(-5, 5, size=(1000, 1))
        # Normalize the features
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        y = np.sin(X) + 0.1 * np.random.randn(1000, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=sk_seed)

        # Neural network hyperparameters
        input_dim = X_train.shape[1]
        hidden_dim = 50
        output_dim = 1
        num_epochs = 100
        learning_rate = 0.01

        # Sample from the posterior using different samplers
        num_samples = 1000
        step_size = 0.1

        # Train the neural network
        model_lmc = TwoLayerNN(input_dim, hidden_dim, output_dim, X_train, y_train, X_test, y_test, task='regression')
        model_ulmc = TwoLayerNN(input_dim, hidden_dim, output_dim, X_train, y_train, X_test, y_test, task='regression')

        lmc_sampler = LangevinMonteCarlo(model_lmc.potential_func, step_size, num_samples)
        ulmc_sampler = UnderdampedLangevinMonteCarlo(model_ulmc.potential_func, step_size, num_samples, gamma=1.0)

        model_lmc.set_sampler(lmc_sampler)
        model_ulmc.set_sampler(ulmc_sampler)

        model_lmc.train(num_epochs, learning_rate)
        model_ulmc.train(num_epochs, learning_rate)

        lmc_params = model_lmc.sample()
        ulmc_params = model_ulmc.sample()

        # Make predictions using the sampled parameters
        lmc_predictions = model_lmc.predict(lmc_params)
        ulmc_predictions = model_ulmc.predict(ulmc_params)


        # Evaluate accuracy
        lmc_accuracy = accuracy_score(y_test, lmc_predictions)
        ulmc_accuracy = accuracy_score(y_test, ulmc_predictions)

        # Plot the predictive function for the two-layer neural network
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test, y_test, c='b', label='True Labels')
        plt.plot(X_test, lmc_predictions, 'r', label='LMC Predictions')
        plt.plot(X_test, ulmc_predictions, 'g', label='ULMC Predictions')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.title('Two-Layer Neural Network - Predictive Function')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig('nn.pdf', format='pdf', dpi=600)
    print('Done')

'''
# Neural network hyperparameters
input_dim = X_train_tensor.shape[1]
hidden_dim = 50
output_dim = 1
num_epochs = 100
learning_rate = 0.01

# Train the neural network
model = TwoLayerNN(input_dim, hidden_dim, output_dim)
train_nn(model, X_train_tensor, y_train_tensor, num_epochs, learning_rate)

# Sample from the posterior using different samplers
num_samples = 1000
step_size = 0.1

lmc_sampler = LangevinMonteCarlo(potential_func_nn, step_size, num_samples)
mala_sampler = MetropolisAdjustedLangevinAlgorithm(potential_func_nn, step_size, num_samples)
hmc_sampler = HamiltonianMonteCarlo(potential_func_nn, step_size, num_samples, leapfrog_steps=10)

lmc_params = sample_nn_posterior(lmc_sampler, num_samples)
mala_params = sample_nn_posterior(mala_sampler, num_samples)
hmc_params = sample_nn_posterior(hmc_sampler, num_samples)

# Make predictions using the sampled parameters
lmc_predictions = predict_nn(X_test, lmc_params)
mala_predictions = predict_nn(X_test, mala_params)
hmc_predictions = predict_nn(X_test, hmc_params)

# Evaluate accuracy
lmc_accuracy = accuracy_score(y_test, lmc_predictions)
mala_accuracy = accuracy_score(y_test, mala_predictions)
hmc_accuracy = accuracy_score(y_test, hmc_predictions)








# Instantiate the samplers
num_samples = 1000
step_size = 0.1
num_features = X_train_tensor.shape[1]

lmc_sampler = LangevinMonteCarlo(num_features, step_size, num_samples)
mala_sampler = MetropolisAdjustedLangevinAlgorithm(num_features, step_size, num_samples)
hmc_sampler = HamiltonianMonteCarlo(num_features, step_size, num_samples, leapfrog_steps=10)

# Bayesian logistic regression with different samplers
blr_lmc = BayesianLogisticRegression(num_features, num_samples, lmc_sampler)
blr_mala = BayesianLogisticRegression(num_features, num_samples, mala_sampler)
blr_hmc = BayesianLogisticRegression(num_features, num_samples, hmc_sampler)

# Sample weights
lmc_weights = blr_lmc.sample_weights()
mala_weights = blr_mala.sample_weights()
hmc_weights = blr_hmc.sample_weights()

# Make predictions
lmc_predictions = blr_lmc.predict(X_test, lmc_weights)
mala_predictions = blr_mala.predict(X_test, mala_weights)
hmc_predictions = blr_hmc.predict(X_test, hmc_weights)

# Evaluate accuracy
lmc_accuracy = accuracy_score(y_test, lmc_predictions)
mala_accuracy = accuracy_score(y_test, mala_predictions)
hmc_accuracy = accuracy_score(y_test, hmc_predictions)

print("Bayesian Logistic Regression Accuracy:")
print("LMC: {:.4f}".format(lmc_accuracy))
print("MALA: {:.4f}".format(mala_accuracy))
print("HMC: {:.4f}".format(hmc_accuracy))

'''