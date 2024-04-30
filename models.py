import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class BayesianLogisticRegression:
    def __init__(self, num_features, num_samples, x_train, y_train, x_test, y_test):
        self.num_features = num_features
        self.num_samples = num_samples
        self.sampler = None
        self.prior = 'gaussian'
        self.x_train = x_train 
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.prior_weight = 1.0
        self.x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        self.x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        self.y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    def set_sampler(self, sampler):
        self.sampler = sampler

    def set_prior(self, prior, prior_weight=1.0):
        assert prior in ['gaussian', 'laplace']
        self.prior = prior

    def potential_func(self, weights):
        logits = torch.matmul(self.x_train_tensor, weights)
        y1 = self.y_train_tensor * torch.log(torch.sigmoid(logits))
        y0 = (1 - self.y_train_tensor) * torch.log(1 - torch.sigmoid(logits))
        log_likelihood = torch.sum(y1 + y0)
        if self.prior == 'gaussian':
            log_prior = -0.5 * torch.sum(weights ** 2)  
        elif self.prior == 'laplace':
            log_prior = -torch.sum(torch.abs(weights))
        else:
            raise ValueError("Prior must be either 'gaussian' or 'laplace'")
        return -log_likelihood - (self.prior_weight *log_prior)

    def sample(self, initial_velocity=None):
        initial_weights = torch.zeros(self.num_features, 1)
        if initial_velocity is not None:
            weights_samples = self.sampler.sample(initial_weights, initial_velocity)
        else:
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
        return y_pred_mean

class TwoLayerNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, x_train, y_train, x_test, y_test, task='classification'):
        super(TwoLayerNN, self).__init__()
        self.hidden_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.task = task
        self.sampler = None
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

    def set_sampler(self, sampler):
        self.sampler = sampler

    def set_prior(self, prior):
        assert prior in ['gaussian', 'laplace']
        self.prior = prior

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
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    def potential_func(self, params):
        model_state_dict = self.state_dict()
        param_dict = {key: params[i] for i, key in enumerate(model_state_dict.keys())}
        self.load_state_dict(param_dict)

        outputs = self(self.x_train_tensor)
        if self.task == 'classification':
            log_likelihood = -torch.nn.BCELoss()(outputs, self.y_train_tensor)
        else:
            log_likelihood = -torch.nn.MSELoss()(outputs, self.y_train_tensor)
        log_prior = -0.5 * sum(param.pow(2).sum() for param in self.parameters()) # Gaussian prior with zero mean and unit variance
        return -(log_likelihood + log_prior)
    
    def sample(self):
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
