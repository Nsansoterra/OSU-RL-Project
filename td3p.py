import numpy as np
import cv2
import functools
from dm_control.locomotion import soccer as dm_soccer
from absl import app, flags
import PIL.Image

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

avg_losses = []
avg_actor_losses = []

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

def plot_avg_loss(show_result=False):
    plt.figure(3)
    plt.clf()
    plt.title('Average Loss (Critic vs Actor)')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    loss_t = torch.tensor(avg_losses, dtype=torch.float)
    actor_loss_t = torch.tensor(avg_actor_losses, dtype=torch.float)

    plt.plot(loss_t.numpy(), label='Critic Loss')
    plt.plot(actor_loss_t.numpy(), label='Actor Loss')

    if len(loss_t) >= 20:
        critic_means = loss_t.unfold(0, 20, 1).mean(1).view(-1)
        critic_means = torch.cat((torch.zeros(19), critic_means))
        plt.plot(critic_means.numpy(), label='Critic Loss (Smoothed)')

    if len(actor_loss_t) >= 20:
        actor_means = actor_loss_t.unfold(0, 20, 1).mean(1).view(-1)
        actor_means = torch.cat((torch.zeros(19), actor_means))
        plt.plot(actor_means.numpy(), label='Actor Loss (Smoothed)')

    plt.legend()
    plt.pause(0.001)

    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def plot_avg_reward(show_result=False):
    plt.figure(2)
    rewards_t = torch.tensor(total_rewards, dtype=torch.float)
    if show_result:
        plt.title('Reward')
    else:
        plt.clf()
        plt.title('Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 384)
        self.fc2 = nn.Linear(384, 384)
        self.fc3 = nn.Linear(384, action_size)
        self.max_action = max_action
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.max_action
        return x
    
    
class Critic(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, 384)
        self.fc2 = nn.Linear(384, 384)
        self.fc3 = nn.Linear(384, 1)

        self.fc4 = nn.Linear(state_size + action_size, 384)
        self.fc5 = nn.Linear(384, 384)
        self.fc6 = nn.Linear(384, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1
    
class SumTree:
    """
    A binary tree data structure where the parent's value is the sum of its children
    Used for efficient sampling in Prioritized Experience Replay
    """
    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes
        self.tree = np.zeros(2 * capacity - 1)  # Total nodes in a binary tree: 2n-1
        self.data_pointer = 0  # Points to the next available leaf node

    def update(self, idx, priority):
        """Update the priority of a leaf node and propagate the change up the tree"""
        # Convert to tree index (leaf nodes start at index capacity-1)
        tree_idx = idx + self.capacity - 1
        
        # Update the leaf node
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # Propagate the change through the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, priority, data):
        """Add a new experience to the tree with its priority"""
        # Get the index in data array
        idx = self.data_pointer
        
        # Update the tree
        self.update(idx, priority)
        
        # Circular buffer for data
        self.data_pointer = (self.data_pointer + 1) % self.capacity

        return idx

    def get(self, s):
        """
        Get a leaf node using a value s (0 <= s <= total_priority)
        Returns (tree_idx, priority, data_idx)
        """
        parent_idx = 0
        
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # If we reach a leaf node
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
                
            # Otherwise, descend to the appropriate child
            if s <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                s -= self.tree[left_child_idx]
                parent_idx = right_child_idx
                
        data_idx = leaf_idx - (self.capacity - 1)
        
        return leaf_idx, self.tree[leaf_idx], data_idx

    def total_priority(self):
        """Return the sum of all priorities (root node)"""
        return self.tree[0]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer implementation
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        self.tree = SumTree(capacity)
        self.capacity = capacity  # Max number of transitions to store
        self.memory = [None] * capacity  # Actual storage for transitions
        self.alpha = alpha  # How much prioritization to use (0 = no prioritization, 1 = full prioritization)
        self.beta = beta  # Importance-sampling correction factor (starts low, increases to 1)
        self.beta_increment = beta_increment  # How much to increase beta each time
        self.epsilon = epsilon  # Small constant to add to priorities to ensure non-zero
        self.max_priority = 1.0  # Max priority to use for new transitions
        
    def push(self, transition):
        """Add a new transition to the buffer with max priority"""
        # Use max_priority for new transitions to encourage exploration
        priority = self.max_priority ** self.alpha
        idx = self.tree.add(priority, None)  # We don't need to store the actual data in the tree
        self.memory[idx] = transition
        
    def sample(self, batch_size):
        """Sample a batch of transitions based on their priorities"""
        batch = []
        indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)
        
        # Calculate the segment size
        segment = self.tree.total_priority() / batch_size
        
        # Increase beta each time we sample
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get min priority for normalization (excluding zero priorities)
        non_zero_priorities = [p for p in self.tree.tree[-self.capacity:] if p > 0]
        if non_zero_priorities:
            min_priority = min(non_zero_priorities) / self.tree.total_priority()
        else:
            min_priority = self.epsilon
            
        # Sample from each segment
        for i in range(batch_size):
            # Get a value within the segment
            a, b = segment * i, segment * (i + 1)
            value = np.random.uniform(a, b)
            
            # Get the corresponding leaf node and data index
            leaf_idx, priority, data_idx = self.tree.get(value)
            
            # Normalize the priority to get the sampling probability
            sampling_prob = priority / self.tree.total_priority()
            
            # Calculate importance sampling weight
            weight = (sampling_prob * self.capacity) ** (-self.beta)
            
            # Normalize weights
            weights[i] = weight / ((min_priority * self.capacity) ** (-self.beta))
            indices[i] = data_idx
            
            # Get the transition
            batch.append(self.memory[data_idx])
            
        weights_tensor = torch.tensor(weights, dtype=torch.float, device=device)
        return batch, indices, weights_tensor
    
    def update_priorities(self, indices, priorities):
        """Update priorities for indices based on TD errors"""
        for idx, priority in zip(indices, priorities):
            # Add a small constant to avoid zero priority
            priority = (priority + self.epsilon) ** self.alpha
            
            # Update max priority for new transitions
            self.max_priority = max(self.max_priority, priority)
            
            # Update the priority in the tree
            self.tree.update(idx, priority)
            
    def __len__(self):
        """Return the current size of the buffer"""
        return min(self.capacity, sum(1 for x in self.memory if x is not None))
    
class RunningMeanStd:
    """
    Tracks the mean and variance of a data stream (e.g., states).
    Useful for normalizing observations in RL.
    """
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # small constant to avoid division by zero

    def update(self, x):
        """
        Update running mean/std with a batch of observations x (shape = [batch_size, num_features]).
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """
        Given mean, var, and count of a batch, update the global mean/var/count.
        """
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # New mean is a weighted average
        new_mean = self.mean + delta * (batch_count / total_count)

        # For variance, we use the one-pass update formula
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * (self.count * batch_count / total_count)
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        """
        Normalize a batch of observations x using the current mean and variance.
        """
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class TD3:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        lr=1e-4,
        exploration_noise=0.3
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.exploration_noise = exploration_noise  # Store the exploration noise
        self.exploration_decay = 0.9999  # Decay factor for exploration noise
        self.total_it = 0

    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
            if not evaluate:
                # Add explicit exploration noise that decays over time
                noise = np.random.normal(0, self.exploration_noise, size=action.shape)
                action = action + noise
                
                # Decay exploration noise
                self.exploration_noise = max(0.05, self.exploration_noise * self.exploration_decay)
                
            return np.clip(action, -self.max_action, self.max_action)

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        if len(replay_buffer) < batch_size:
            return None, None

        # Sample from replay buffer with priorities
        batch, indices, weights = replay_buffer.sample(batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)

        states = torch.cat(states)
        actions = torch.cat(actions)
        next_states = torch.cat(next_states)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        dones = torch.tensor(dones, dtype=torch.float, device=device)

        with torch.no_grad():
            noise = (
                torch.randn_like(actions) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_actions = (
                self.actor_target(next_states) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute target Q value
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.discount * target_Q

            target_Q = torch.clamp(target_Q, -1e6, 1e6)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(states, actions)

        # Compute TD errors for updating priorities
        td_error1 = torch.abs(current_Q1 - target_Q).detach().cpu().numpy()
        td_error2 = torch.abs(current_Q2 - target_Q).detach().cpu().numpy()
        td_errors = np.mean([td_error1, td_error2], axis=0).flatten()
        
        # Update priorities in the replay buffer
        replay_buffer.update_priorities(indices, td_errors)

        # Compute critic loss with importance sampling weights
        critic_loss = (weights * F.smooth_l1_loss(current_Q1, target_Q, reduction='none')).mean() + \
                     (weights * F.smooth_l1_loss(current_Q2, target_Q, reduction='none')).mean()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        actor_loss = None

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # Update the target networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if actor_loss is not None:
            actor_loss = actor_loss.item()
        
        return critic_loss.item(), actor_loss


def process_observation(timestep, player_idx):
    """
    Process the observation from a timestep for a specific player.
    
    Args:
        timestep: The environment timestep containing observations
        player_idx: The index of the player (0 for the first player)
        
    Returns:
        A numpy array containing the flattened observation
    """
    # Get the observation dictionary for the specified player
    obs_dict = timestep.observation[player_idx]

    ball_vel = obs_dict.get('stats_vel_to_ball', None)
    goal_vel = obs_dict.get('stats_vel_ball_to_goal', None)
    
    # Collect and flatten all observations
    obs_components = []
    for key, value in obs_dict.items():
        if isinstance(value, np.ndarray):
            obs_components.append(value.flatten())
        else:
            # Convert non-array values to a numpy array for consistency
            obs_components.append(np.array([value]))

    
    flat_obs = np.concatenate(obs_components)
    return flat_obs, ball_vel, goal_vel

# Then, to create a tensor for your network:
def get_observation_tensor(timestep, player_idx=0, device='cuda'):
    """
    Convert observation to a tensor ready for the neural network.
    """
    flat_obs, ball_vel, goal_vel = process_observation(timestep, player_idx)

    obs_normalizer.update(flat_obs[np.newaxis, :])  

    normalized_obs = obs_normalizer.normalize(flat_obs)

    obs_tensor = torch.tensor(normalized_obs, dtype=torch.float32).unsqueeze(0).to(device)
    
    return obs_tensor, ball_vel, goal_vel


# Hyperparameters
BATCH_SIZE = 256
GAMMA = 0.995  # discount factor
TAU = 0.005  # for target network update
LR = 1e-4  # learning rate 
POLICY_NOISE = 0.2  # noise added to target policy during critic update
NOISE_CLIP = 0.5  # limit for noise
POLICY_FREQ = 2  # frequency of delayed policy updates
PER_ALPHA = 0.6  # Priority exponent
PER_BETA = 0.4  # Initial importance sampling weight
PER_BETA_INCREMENT = 0.0001  # How much to increase beta each time



# Set up flags (for any optional arguments)
FLAGS = flags.FLAGS
flags.DEFINE_enum("walker_type", "BOXHEAD", ["BOXHEAD", "ANT", "HUMANOID"], "The type of walker to explore with.")
flags.DEFINE_bool("enable_field_box", True, "Enable the physical bounding box enclosing the ball.")
flags.DEFINE_bool("disable_walker_contacts", False, "Disable walker-walker contacts.")
flags.DEFINE_bool("terminate_on_goal", False, "Terminate on goal.")

# Initialize the soccer environment with random actions
random_state = np.random.RandomState(42)
env = dm_soccer.load(team_size=1,
                     time_limit=50000.0,
                     disable_walker_contacts=False,
                     keep_aspect_ratio=True,
                     enable_field_box=True,
                     terminate_on_goal=False,
                     walker_type=dm_soccer.WalkerType.BOXHEAD)





player_action_spec = env.action_spec()[0]  
action_dim = player_action_spec.shape[0]  # 3 for soccer environment
max_action = float(player_action_spec.maximum[0])  # Assuming symmetric action space



state_dim = 77
agent = TD3(
    state_dim=state_dim,
    action_dim=action_dim,
    max_action=max_action,
    discount=GAMMA,
    tau=TAU,
    policy_noise=POLICY_NOISE * max_action,
    noise_clip=NOISE_CLIP * max_action,
    policy_freq=POLICY_FREQ,
    lr=LR
)
obs_normalizer = RunningMeanStd(shape=(state_dim,))
memory = PrioritizedReplayBuffer(capacity=200000, alpha=PER_ALPHA, beta=PER_BETA, beta_increment=PER_BETA_INCREMENT)

FRAME_SKIP = 4


# Retrieve action_specs for all players
action_specs = env.action_spec()

fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Step through the environment with random actions and capture video
steps_done = 0

episode_durations = []
total_rewards = []

ball_vel_history = deque(maxlen=10)  # Store last 10 velocities toward ball
goal_vel_history = deque(maxlen=10)  # Store last 10 velocities of ball toward goal
distance_history = deque(maxlen=10)  

for episode in count():
    timestep = env.reset()
    distance_history.clear()
    ball_vel_history.clear()
    goal_vel_history.clear()
    state, prev_ball_vel, prev_goal_vel = get_observation_tensor(timestep, player_idx=0, device=device)
    
    total_reward = 0
    stagnant = 0
    proximity_count = 0
    time_since_touch = 0
    min_distance = None

    episode_losses = []
    episode_losses2 = []
    

    if episode % 500 == 0:
        torch.save(agent.actor.state_dict(), f"td3_actor_model_{episode}.pth")
        torch.save(agent.critic.state_dict(), f"td3_critic_model_{episode}.pth")
        torch.save(agent.actor_target.state_dict(), f"td3_actor_target_model_{episode}.pth")
        torch.save(agent.critic_target.state_dict(), f"td3_critic_target_model_{episode}.pth")
    for t in count():
        # Take random actions for each player
        actions = []
        done = False 

        
        continuous_action = agent.select_action(state)
        actions.append(continuous_action)  
        actions.append(np.random.uniform(-1.0, 1.0, size=action_specs[0].shape))
        #print(actions)
        
        frame_reward = 0
        for _ in range(FRAME_SKIP):
            stagnant+=1
            time_since_touch+=1
            steps_done+=1
            timestep = env.step(actions)
            distance_to_ball = np.linalg.norm(timestep.observation[0]['ball_ego_position'])


            #ball_vel is velocity to the ball
            #goal_vel is ball's velocity to goal
            new_state, ball_vel, goal_vel = get_observation_tensor(timestep, player_idx=0, device=device)
            if t > 10:
                goal_vel_history.append(goal_vel)
            distance_history.append(distance_to_ball)

            reward = timestep.reward
            reward = reward[0]
            
            #print("states shape:", state.shape)
            #print("actions shape", action.shape)
            #print("policy_net(states).shape:", policy_net(state).shape)

            if steps_done % 10000 == 0:
                print(f"Steps done: {steps_done}")

            camera_id = 4
            pixels = env.physics.render(camera_id=camera_id, width=640, height=480)
            if steps_done % 4 == 0:
                image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                cv2.imshow('Soccer', image)        

            
            #~~~~~~~~~~~~~calculate reward~~~~~~~~~~~~~#
            #if the reward is > 0, that means a goal has been scored and we reset the environment
            if reward > 0:
                #done = True
                reward+=100

            #if it hasn't scored a goal within the time limit we just reset
            if stagnant > 3000:
                done = True
                break

            #if it hasn't touched the ball within a certain time limit, we reset earlier
            if time_since_touch > 300:
                done = True
                break

            

            frame_reward += reward

        additional_reward = 0


        #one time reward for closing distance to ball
        if min_distance is None:
            min_distance = round(distance_to_ball, 2)
        else:
            current_distance = round(distance_to_ball, 2)
            if current_distance < min_distance:
                improvement = min_distance - current_distance
                min_distance = current_distance
                time_since_touch = 0

            
        if len(distance_history) == distance_history.maxlen:
            past_avg = np.mean(list(distance_history)[:3])  # Avg of earliest 3
            recent_avg = np.mean(list(distance_history)[-2:])  # Avg of most recent 2

            distance_delta = past_avg - recent_avg  # Positive = improvement (closer to ball)
            
            # Reward proportional to improvement
            additional_reward += distance_delta * 5.0  # Scale as needed

            # Optional: clip to avoid extreme values
            additional_reward = np.clip(additional_reward, -1.0, 1.0)


        #reward based on hitting the ball toward the goal
        if len(goal_vel_history) >= 5:
            # Calculate if goal velocity is consistently improving
            goal_vel_trend = sum(goal_vel_history[-1] - goal_vel_history[i] for i in range(-5, -1)) / 4
            if goal_vel_trend > 0.05:
                additional_reward += 3  # Larger reward for moving ball toward goal
                time_since_touch = -100
                distance_history.clear()
            elif goal_vel_trend < -0.05:
                additional_reward += 1  
                time_since_touch = -100
                distance_history.clear()
                


        frame_reward += additional_reward
        total_reward += frame_reward
        reward = frame_reward
        #~~~~~~~~~~~~~calculate reward~~~~~~~~~~~~~#
        #print(reward)
        
        action_tensor = torch.tensor(continuous_action, dtype=torch.float32).unsqueeze(0).to(device)
        r = torch.tensor(reward, dtype=torch.float, device=device)
        memory.push((state, action_tensor, new_state, r, done))
        loss1, loss2 = agent.train(memory, BATCH_SIZE)

        if loss1 is not None:
            episode_losses.append(loss1)
        if loss2 is not None:
            episode_losses2.append(loss2)

        state = new_state
        
        if done:
            total_rewards.append(total_reward)
            if episode_losses:
                avg_loss = np.mean(episode_losses)
                avg_losses.append(avg_loss)
            if episode_losses2:
                avg_actor_loss = np.mean(episode_losses2)
                avg_actor_losses.append(avg_actor_loss)
            plot_avg_reward()
            plot_avg_loss()
            break





        if cv2.waitKey(1) & 0xFF == ord('q'):
            torch.save(agent.actor.state_dict(), f"td3_actor_model_final.pth")
            torch.save(agent.critic.state_dict(), f"td3_critic_model_final.pth")
            torch.save(agent.actor_target.state_dict(), f"td3_actor_target_model_final.pth")
            torch.save(agent.critic_target.state_dict(), f"td3_critic_target_model_final.pth")
            exit(1)