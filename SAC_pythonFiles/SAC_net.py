import os
import torch as T
import torch.nn.functional as F
import numpy as np
from .Buffer import ReplayBuffer
from .SAC import Actor, Critic, Value


class Agent():
    def __init__(self, input_dims, action_size, max_action, name, max_size=1000000, batch_size=256, reward_scale=2, tau=0.005, gamma=0.99, alpha=0.0003, beta=0.0003):
        self.tau = tau
        self.gamma = gamma
        self.memory = ReplayBuffer(max_size, input_dims, action_size)
        self.batch_size = batch_size
        self.action_size = action_size

        self.actor = Actor(alpha, input_dims, action_size, max_action, name='actor'+name)
        self.critic_1 = Critic(beta, input_dims, action_size, name='critic_1'+name)
        self.critic_2 = Critic(beta, input_dims, action_size, name='critic_2'+name)
        self.value = Value(beta, input_dims, name='value'+name)
        self.target_value = Value(beta, input_dims, name='Target_val'+name)

        self.scale = reward_scale
        self.update_network_parameters(tau=1)


    def choose_action(self, observation):
        state = T.tensor(np.array(observation), dtype=T.float32).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]
    
    def remember(self, state, action, reward, n_state, done):
        # add MDP tuple to memory
        self.memory.add_to_buffer(state, action, reward, n_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        #target_value_state_dict = dict(target_value_params)
        #value_state_dict = dict(value_params)

        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


        #self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print("\n SAVING MODELS \n")
        self.actor.save_model()
        self.value.save_model()
        self.target_value.save_model()
        self.critic_1.save_model()
        self.critic_2.save_model()

    def load_models(self):
        print("\n LOADING MODELS \n")
        self.actor.load_model()
        self.value.load_model()
        self.target_value.load_model()
        self.critic_1.load_model()
        self.critic_2.load_model()

    def learn(self):
        # only learn if we have enough experience
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, reward, n_state, done = self.memory.sample_buffer(self.batch_size)

        # format data
        reward = T.tensor(reward, dtype=T.float32).to(self.actor.device)
        done = T.tensor(done, dtype=T.float32).to(self.actor.device)
        state = T.tensor(state, dtype=T.float32).to(self.actor.device)
        n_state = T.tensor(n_state, dtype=T.float32).to(self.actor.device)
        action = T.tensor(action, dtype=T.float32).to(self.actor.device)

        # 
        value = self.value(state).view(-1)
        value_ = self.target_value(n_state).view(-1)
        value_ = value_ * (1 - done.to(dtype=T.float32))

        # update steps for the value network
        actions, log_probs = self.actor.sample_normal(state,reparameterize=False)
        log_probs = log_probs.view(-1)
        actions = actions.to(dtype=T.float32)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = (critic_value - log_probs).to(dtype=T.float32)
        value_loss = (0.5*F.mse_loss(value, value_target))
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()


        # update steps for actor
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        actions = actions.to(dtype=T.float32)
        log_probs = log_probs.to(dtype=T.float32)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss).to(dtype=T.float32)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()


        # update Critics
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        reward = T.tensor(reward * self.scale, dtype=T.float32).to(self.actor.device)
        q_hat = reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()



