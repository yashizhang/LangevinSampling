import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class BayesianLogisticRegression:
    def __init__(self, num_features, num_samples, sampler, x_train, y_train, x_test, y_test):
        self.num_features = num_features
        self.num_samples = num_samples
        self.sampler = sampler
        self.x_train = x_train 
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        self.x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        self.y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    def potential_func(self, weights):
        logits = torch.matmul(self.x_train_tensor, weights)
        log_likelihood = torch.sum(self.y_train_tensor * torch.log(torch.sigmoid(logits)) +
                                   (1 - self.y_train_tensor) * torch.log(1 - torch.sigmoid(logits)))
        log_prior = -0.5 * torch.sum(weights ** 2)  # Gaussian prior with zero mean and unit variance
        return -log_likelihood - log_prior

    def sample_weights(self):
        initial_weights = torch.zeros(self.num_features, 1)
        weights_samples = self.sampler.sample(initial_weights)
        return weights_samples

    def predict(self, X, weights_samples):
        num_samples = len(weights_samples)
        y_pred_samples = np.zeros((num_samples, len(X)))
        for i, weights in enumerate(weights_samples):
            logits = np.dot(X, weights)
            y_pred = 1 / (1 + np.exp(-logits))
            y_pred_samples[i] = y_pred.flatten()
        y_pred_mean = np.mean(y_pred_samples, axis=0)
        y_pred_mean = np.round(y_pred_mean).astype(int)
        return y_pred_mean

class TwoLayerNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, sampler, x_train, y_train, x_test, y_test, task='classification'):
        super(TwoLayerNN, self).__init__()
        self.sampler = sampler
        self.hidden_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.task = task
        self.x_train = x_train 
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        self.x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        self.y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
        if self.task == 'classification':
            self.output_activation = torch.sigmoid
        elif self.task == 'regression':
            self.output_activation = torch.nn.Identity()
        else:
            raise ValueError("Task must be either 'classification' or 'regression'")

    def forward(self, x):        
        x = torch.relu(self.hidden_layer(x))
        x = self.output_activation(self.output_layer(x))
        return x

    def train(self, num_epochs, learning_rate):
        criterion = torch.nn.BCELoss() if self.task == 'classification' else torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = self(self.x_train_tensor)
            loss = criterion(outputs, self.y_train_tensor)
            loss.backward()
            optimizer.step()

    def potential_func(self, params):
        model_state_dict = self.state_dict()
        param_dict = {key: params[i] for i, key in enumerate(model_state_dict.keys())}
        self.load_state_dict(param_dict)

        outputs = self(self.x_train_tensor)
        if self.task == 'classification':
            log_likelihood = -torch.nn.BCELoss()(outputs, self.y_train_tensor)
        else:
            log_likelihood = -torch.nn.MSELoss()(outputs, self.y_train_tensor)
        log_prior = -0.5 * sum(param.pow(2).sum() for param in self.parameters())
        return -(log_likelihood + log_prior)
    
    def sample_weights(self): ####
        num_params = sum(p.numel() for p in self.parameters())
        initial_params = torch.zeros(num_params)
        params_samples = self.sampler.sample(initial_params)
        return params_samples
    
    def predict(self, params_samples):
        num_samples = len(params_samples)
        y_pred_samples = np.zeros((num_samples, len(self.x_test_tensor)))
        for i, params in enumerate(params_samples):
            model_state_dict = self.state_dict()
            param_dict = {key: params[i] for i, key in enumerate(model_state_dict.keys())}
            self.load_state_dict(param_dict)
            with torch.no_grad():
                outputs = self(self.x_test_tensor)
                y_pred = outputs.numpy().flatten()
            y_pred_samples[i] = y_pred
        y_pred_mean = np.mean(y_pred_samples, axis=0)
        if self.task == 'classification':
            y_pred_mean = np.round(y_pred_mean).astype(int)
        y_pred_std = np.std(y_pred_samples, axis=0)
        return y_pred_mean, y_pred_std

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

# Plot the predictive function for Bayesian logistic regression
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolors='k')
plt.colorbar(label='True Labels')
plt.contourf(xx, yy, lmc_predictions.reshape(xx.shape), cmap='RdBu_r', alpha=0.3, levels=20)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Bayesian Logistic Regression - Predictive Function (LMC)')
plt.tight_layout()
plt.show()

# Plot the predictive function for the two-layer neural network
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolors='k')
plt.colorbar(label='True Labels')
plt.contourf(xx, yy, hmc_predictions.reshape(xx.shape), cmap='RdBu_r', alpha=0.3, levels=20)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Two-Layer Neural Network - Predictive Function (HMC)')
plt.tight_layout()
plt.show()
'''