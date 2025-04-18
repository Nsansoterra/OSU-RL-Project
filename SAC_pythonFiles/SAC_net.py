import os
import torch as T
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from .Buffer import ReplayBuffer
from .SAC import Actor, Critic, Value


class Agent():
    def __init__(self, input_dims, action_size, max_action, name, policy_freq=2, max_size=500000, batch_size=256, reward_scale=2, tau=0.005, gamma=0.99, alpha=0.0003, beta=0.0003):
        self.tau = tau
        self.gamma = gamma
        self.memory = ReplayBuffer(max_size, input_dims, action_size)
        self.batch_size = batch_size
        self.action_size = action_size
        self.policy_freq = policy_freq

        self.target_entropy = -np.prod((action_size,)).item()  # heuristic
        self.log_alpha = T.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        self.actor = Actor(alpha, input_dims, action_size, max_action, name='actor'+name)
        self.critic_1 = Critic(beta, input_dims, action_size, name='critic_1'+name)
        self.critic_2 = Critic(beta, input_dims, action_size, name='critic_2'+name)
        self.value = Value(beta, input_dims, name='value'+name)
        self.target_value = Value(beta, input_dims, name='Target_val'+name)

         # Normalization stats
        self.obs_mean = np.zeros(input_dims, dtype=np.float32)
        self.obs_std = np.ones(input_dims, dtype=np.float32)
        self.obs_count = 1e-4  # prevent division by zero

        # Normalization stats for rewards
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 1e-4

        self.scale = reward_scale
        self.total_step = 0


    def update_reward_stats(self, reward_batch):
        batch_mean = np.mean(reward_batch)
        batch_var = np.var(reward_batch)
        batch_count = reward_batch.shape[0]

        total_count = self.reward_count + batch_count
        delta = batch_mean - self.reward_mean

        new_mean = self.reward_mean + delta * batch_count / total_count
        m_a = self.reward_std ** 2 * self.reward_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.reward_count * batch_count / total_count
        new_var = M2 / total_count

        self.reward_mean = new_mean
        self.reward_std = np.sqrt(new_var + 1e-6)
        self.reward_count = total_count


    def update_obs_stats(self, obs_batch):
        # Running mean and variance update
        batch_mean = np.mean(obs_batch, axis=0)
        batch_var = np.var(obs_batch, axis=0)
        batch_count = obs_batch.shape[0]

        total_count = self.obs_count + batch_count
        delta = batch_mean - self.obs_mean

        new_mean = self.obs_mean + delta * batch_count / total_count
        m_a = self.obs_std ** 2 * self.obs_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.obs_count * batch_count / total_count
        new_var = M2 / total_count

        self.obs_mean = new_mean
        self.obs_std = np.sqrt(new_var + 1e-6)
        self.obs_count = total_count

    def normalize_obs(self, obs):
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)


    def choose_action(self, observation):
        obs = self.normalize_obs(np.asarray(observation))
        state = T.tensor(np.asarray(obs), dtype=T.float32).to(self.actor.device)
        actions, _ = self.actor.forward(state)

        return actions.cpu().detach().numpy()
    
    def remember(self, state, action, reward, n_state, done):
        # add MDP tuple to memory
        self.memory.add_to_buffer(state, action, reward, n_state, done)

    

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



    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau

        for t_param, l_param in zip(self.target_value.parameters(), self.value.parameters()):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)



    def learn(self):
        # only learn if we have enough experience
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, reward, n_state, done = self.memory.sample_buffer(self.batch_size)

         # Update normalization stats and normalize
        self.update_obs_stats(state)
        self.update_reward_stats(reward)
        state = self.normalize_obs(state)
        n_state = self.normalize_obs(n_state)
        reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        

        # format data
        reward = T.tensor(reward, dtype=T.float32).to(self.actor.device)
        done = T.tensor(done, dtype=T.float32).to(self.actor.device)
        state = T.tensor(state, dtype=T.float32).to(self.actor.device)
        n_state = T.tensor(n_state, dtype=T.float32).to(self.actor.device)
        action = T.tensor(action, dtype=T.float32).to(self.actor.device)

        new_action, log_prob = self.actor.forward(state)

        alpha_loss = (-self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()  # used for the actor loss calculation

        # 
        value = self.value.forward(state).view(-1)
        value_ = self.target_value(n_state).view(-1)
        value_ = value_ * (1 - done.to(dtype=T.float32))

        # update steps for the value network
        log_prob = log_prob.view(-1)
        new_action = new_action.to(dtype=T.float32)
        q1_new_policy = self.critic_1.forward(state, new_action)
        q2_new_policy = self.critic_2.forward(state, new_action)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        value_target = (critic_value - log_prob*alpha).to(dtype=T.float32)
        value_loss = (F.mse_loss(value, value_target))
        self.value.optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()
        

        if self.total_step % self.policy_freq == 0:
            advantage = critic_value
            actor_loss = T.mean((alpha*log_prob - advantage))

            self.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor.optimizer.step()

            self._target_soft_update()
        else:
            actor_loss = T.zeros(())


        # update Critics
        log_prob = log_prob.view(-1)
        new_action = new_action.to(dtype=T.float32)
        q1_new_policy = self.critic_1.forward(state, new_action)
        q2_new_policy = self.critic_2.forward(state, new_action)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        reward = T.tensor(reward * self.scale, dtype=T.float32).to(self.actor.device)
        q_hat = reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = F.mse_loss(q2_old_policy, q_hat)

        self.critic_1.optimizer.zero_grad()
        critic_1_loss.backward(retain_graph=True)
        self.critic_1.optimizer.step()

        self.critic_2.optimizer.zero_grad()
        critic_2_loss.backward(retain_graph=True)
        self.critic_2.optimizer.step()

        
        self.total_step += 1

        return critic_1_loss.detach().numpy(), critic_2_loss.detach().numpy(), actor_loss.detach().numpy(), value_loss.detach().numpy()
        



