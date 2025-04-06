import torch as T
import torch.nn as nn
import os
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np



# Critic Network
class Critic(nn.Module):
    def __init__(self, beta, input_dim, output_dim, name="critic", savepath="../SAC_nets"):
        super(Critic, self).__init__()
        # save parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.savepath = savepath
        if not os.path.exists(savepath):
            os.makedirs(savepath)  # Create the directory if it doesn't exist
        self.savepoint_file = os.path.join(self.savepath, name+"_sac")

        self.fc1 = nn.Linear(self.input_dim+output_dim, 256)
        self.fc2 = nn.Linear(256,256)
        self.q = nn.Linear(256, 1)

        self.optimizer = optim.Adam(self.parameters(), lr = beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        # standard NN forward pass using relu
        action_val = self.fc1(T.cat([state.float(), action.float()], dim=1))
        action_val = F.relu(action_val)
        action_val = self.fc2(action_val)
        action_val = F.relu(action_val)

        q = self.q(action_val)

        return q
    
    def save_model(self):
        # save the model
        T.save(self.state_dict(), self.savepoint_file)

    def load_model(self):
        # load the model
        self.load_state_dict(T.load(self.savepoint_file))


class Value(nn.Module):
    def __init__(self, beta, input_dims, name='value', savepath='../SAC_nets'):
        super(Value, self).__init__()
        self.input_dims = input_dims
        self.name = name
        self.savepath = savepath
        if not os.path.exists(savepath):
            os.makedirs(savepath)  # Create the directory if it doesn't exist
        self.savepath_file = os.path.join(self.savepath, name+'_sac')

        self.fc1 = nn.Linear(self.input_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.v = nn.Linear(256, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        state_value = F.relu(self.fc1(state))
        state_value = F.relu(self.fc2(state_value))
        
        v = self.v(state_value)

        return v
    
    def save_model(self):
        T.save(self.state_dict(), self.savepath_file)

    def load_model(self):
        self.load_state_dict(T.load(self.savepath_file))


# Actor Network
class Actor(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, max_action, name="actor", save_path="../SAC_nets"):
        """
        Input Args-

        State_dim: the dimension of the state representation
        Action_dim: dimension of the actions
        max_action: the max action values -> used for scaling
        name: the name of the network ->used for saving and checkpoints
        save_path: the directory of where the networks will be saved

        ."""
        super(Actor, self).__init__()
        # save parameters
        self.save_dir = save_path
        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.savepoint_dir = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)  # Create the directory if it doesn't exist
        self.savepoint_file = os.path.join(self.savepoint_dir, name+'_sac')

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        # 2 different ouput layers
        self.log_std_layer = nn.Linear(256, action_dim)
        self.mu_layer = nn.Linear(256, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    
    def forward(self, state):
        # first two fully connected layers are the same for both ouputs
        state = state.to(self.device)
        # x = self.fc1(state)
        # x = T.relu(x)
        # x = self.fc2(x)
        # x = T.relu(x)

        # # mu and sigma
        # mu = T.tanh(self.mu_layer(x))
        # sigma = self.log_std_layer(x)

        # sigma = T.clamp(sigma,-20, 2)
        # std = T.exp(sigma)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Compute mean and log standard deviation
        mu = self.mu_layer(x).tanh()
        log_std = self.log_std_layer(x).tanh()
        log_std = -20 + 0.5 * (2 + 20) * (log_std + 1)
        std = T.exp(log_std)

        # Action distribution and sampling
        dist = Normal(mu, std)
        z = dist.rsample()
        action = z.tanh()
        log_prob = dist.log_prob(z) - T.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob

        #return mu, std

    # def sample_normal(self, state, reparameterize=True):
    #     mu, sigma = self.forward(state)
    #     probs = Normal(mu,sigma)

    #     if reparameterize:
    #         actions = probs.rsample()
    #     else:
    #         actions = probs.sample()

    #     action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)
    #     log_probs = probs.log_prob(actions) - T.log(1-action.pow(2)+1e-6)
    #     log_probs = log_probs.sum(-1, keepdim=True)

    #     return action, log_probs
    
    def save_model(self):
        T.save(self.state_dict(), self.savepoint_file)
        print(self.state_dict())

    def load_model(self):
        self.load_state_dict(T.load(self.savepoint_file))
        print(self.state_dict())