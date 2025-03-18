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

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Distance')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
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

class NoisyLayer(nn.Module):
    def __init__(self, in_dim, out_dim, sigma_init = 0.017):
        super(NoisyLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight_mu = nn.Parameter(torch.full((out_dim, in_dim), 0.0, dtype=torch.float32))
        self.weight_sigma = nn.Parameter(torch.full((out_dim, in_dim), sigma_init, dtype=torch.float32))
        self.register_buffer('weight_epsilon', torch.zeros(out_dim, in_dim, dtype=torch.float32))

        self.bias_mu = nn.Parameter(torch.full((out_dim,), 0.0, dtype=torch.float32))
        self.bias_sigma = nn.Parameter(torch.full((out_dim,), sigma_init, dtype=torch.float32))
        self.register_buffer('bias_epsilon', torch.zeros(out_dim, dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_dim)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

    def forward(self, x):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon

        return F.linear(x, weight, bias)

class DQN(nn.Module):
    def __init__(self, state_size, n_actions):
        super(DQN, self).__init__()
        # For vector input
        self.fc1 = NoisyLayer(state_size, 384)
        self.fc2 = NoisyLayer(384, 384)
        
        # Value stream
        self.fc1_value = NoisyLayer(384, 384)
        self.fc2_value = NoisyLayer(384, 1)
        
        # Advantage stream
        self.fc1_advantage = NoisyLayer(384, 384)
        self.fc2_advantage = NoisyLayer(384, n_actions)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        value = torch.relu(self.fc1_value(x))
        value = self.fc2_value(value)
        
        advantage = torch.relu(self.fc1_advantage(x))
        advantage = self.fc2_advantage(advantage)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    
class ReplayMemory(object):
    def __init__(self, capacity, n_step=3, gamma=0.99):
        self.memory = deque([], maxlen = capacity)
        self.n_step_buffer = deque([], maxlen=n_step)  # A buffer to store n steps of transitions
        self.n_step = n_step  # Number of steps for multistep learning
        self.gamma = gamma

    def push(self, transition):
        # Add transition to the n-step buffer
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) == self.n_step:
            # Get the n-step reward and future state
            state, action, _, _, _ = self.n_step_buffer[0]
            _, _, next_state, reward, done = self.n_step_buffer[-1]

            n_step_reward = sum(self.gamma**i * self.n_step_buffer[i][3] for i in range(self.n_step))

            self.memory.append((state, action, next_state, n_step_reward, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

def select_action(obs):
    global steps_done
    sample = random.random()
    
    steps_done +=1


    with torch.no_grad():
        # Get the discrete action index
        discrete_action = policy_net(obs).argmax().item()
        q_values = policy_net(obs)
        #print(q_values)
        # Return the discrete action as a tensor
        return torch.tensor([discrete_action], device=device)
    
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    batch = memory.sample(BATCH_SIZE)

    states, actions, new_states, rewards, dones = zip(*batch)

    states = torch.cat(states)  
    actions = torch.cat(actions) 
    actions = actions.long()
    new_states = torch.cat(new_states) 
    rewards = torch.tensor(rewards, dtype=torch.float, device=device)
    dones = torch.tensor(dones, dtype=torch.float, device=device)
    
    with torch.no_grad():
        best_policy_actions = policy_net(new_states).argmax(dim=1)

        target_q = rewards + (1-dones) * (GAMMA ** memory.n_step) * \
            target_net(new_states).gather(dim=1, index=best_policy_actions.unsqueeze(dim=1)).squeeze()

    current_q = policy_net(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
    criterion = nn.SmoothL1Loss()
    loss = criterion(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def discretize_action_space(player_action_spec, bins_per_dimension=3):
    # Get the shape, min, and max values of the continuous action space
    action_dims = player_action_spec.shape[0]  # Should be 3
    action_min = player_action_spec.minimum
    action_max = player_action_spec.maximum
    
    # Calculate total number of discrete actions (3^3 = 27 with bins_per_dimension=3)
    total_discrete_actions = bins_per_dimension ** action_dims
    print(f"Total discrete actions: {total_discrete_actions}")
    
    # Function to convert discrete action index to continuous action vector
    def discrete_to_continuous(action_idx):
        continuous_action = np.zeros(action_dims)
        
        for dim in range(action_dims):
            # Calculate which bin this dimension should use
            bin_size = (action_max[dim] - action_min[dim]) / bins_per_dimension
            bin_idx = (action_idx // (bins_per_dimension ** dim)) % bins_per_dimension
            continuous_action[dim] = action_min[dim] + (bin_idx + 0.5) * bin_size
            
        return continuous_action
    
    return total_discrete_actions, discrete_to_continuous


def process_observation(timestep, player_idx=0):
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
    #print(obs_dict)
    opp_dict = timestep.observation[1]

    ball_vel = obs_dict['stats_vel_to_ball']
    goal_vel = obs_dict['stats_vel_ball_to_goal']
    
    # Select the relevant features (you can modify this list based on what you need)
    relevant_keys = [
        'body_height',
        'joints_pos', 
        'joints_vel',
        'sensors_velocimeter',
        'sensors_gyro',
        'prev_action',
        'sensors_accelerometer',
        'ball_ego_position',
        'ball_ego_linear_velocity',
        'ball_ego_angular_velocity',
        'stats_vel_to_ball',
        'stats_vel_ball_to_goal',
        'stats_home_avg_teammate_dist',
        'stats_home_score',
        'stats_away_score',
        'end_effectors_pos',
        'world_zaxis',
        'team_goal_back_right',
        'team_goal_mid',
        'team_goal_front_left',
        'field_front_left',
        'stats_veloc_forward'
    ]
    
    # Collect and flatten the selected observations
    obs_components = []
    for key in relevant_keys:
        if key in obs_dict:
            obs_components.append(obs_dict[key].flatten())

    obs_components.append(opp_dict['opponent_0_ego_position'].flatten())
    
    # Concatenate all components into a single vector
    flat_obs = np.concatenate(obs_components)
    return flat_obs, ball_vel, goal_vel

# Then, to create a tensor for your network:
def get_observation_tensor(timestep, player_idx=0, device='cuda'):
    """
    Convert observation to a tensor ready for the neural network.
    """
    flat_obs, ball_vel, goal_vel = process_observation(timestep, player_idx)
    
    # Convert to tensor and add batch dimension
    obs_tensor = torch.tensor(flat_obs, dtype=torch.float32).unsqueeze(0).to(device)
    return obs_tensor, ball_vel, goal_vel


BATCH_SIZE = 128
GAMMA = 0.99 #discount factor
TAU = 0.05 #update rate of target network
LR = 0.00025



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
                     enable_field_box=True,
                     terminate_on_goal=False,
                     walker_type=dm_soccer.WalkerType.BOXHEAD)




player_action_spec = env.action_spec()[0]  
n_actions, action_converter = discretize_action_space(player_action_spec, bins_per_dimension=3)

policy_net = DQN(48, n_actions).to(device)
policy_net.load_state_dict(torch.load('dqn_model600.pth'))
target_net = DQN(48, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
policy_net.train()
target_net.train()

optimizer = optim.Adam(policy_net.parameters(), lr=LR, amsgrad=True)

memory = ReplayMemory(100000, n_step=3, gamma=0.99)



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
    state, prev_ball_vel, prev_goal_vel = get_observation_tensor(timestep, player_idx=0, device=device)
    
    total_reward = 0
    stagnant = 0
    proximity_count = 0
    time_since_touch = 0

    if(episode % 100 == 0):
        torch.save(policy_net.state_dict(), f"dqn_model_contin_{episode}.pth")
    for t in count():
        # Take random actions for each player
        actions = []
        done = False 
        stagnant+=1
        time_since_touch+=1
        
        action = select_action(state)  
        continuous_action = action_converter(action.item())  
        actions.append(continuous_action)  
        actions.append(np.random.uniform(-1.0, 1.0, size=action_specs[0].shape))
        #print(actions)
        timestep = env.step(actions)

        player_pos = timestep.observation[1]['opponent_0_ego_position']
        ball_pos = timestep.observation[1]['ball_ego_position']
        distance_to_ball = np.linalg.norm(player_pos[0] - ball_pos[0])

        #ball_vel is velocity to the ball
        #goal_vel is ball's velocity to goal
        new_state, ball_vel, goal_vel = get_observation_tensor(timestep, player_idx=0, device=device)
        ball_vel_history.append(ball_vel)
        goal_vel_history.append(goal_vel)
        distance_history.append(distance_to_ball)


        reward = timestep.reward
        reward = reward[0]*10

        
        #print("states shape:", state.shape)
        #print("actions shape", action.shape)
        #print("policy_net(states).shape:", policy_net(state).shape)

        
        #~~~~~~~~~~~~~calculate reward~~~~~~~~~~~~~#
        #if the reward is > 0, that means a goal has been scored and we reset the environment
        if reward > 0:
            done = True
            reward+=300

        #if it hasn't scored a goal within the time limit we just reset
        if stagnant > 10000:
            done = True
            reward-=50

        #if it hasn't touched the ball within a certain time limit, we reset earlier
        if time_since_touch > 1000:
            done = True
            reward-=50


        additional_reward = 0

        if len(distance_history) >= 5:
            #print(distance_history[0])
            #print(distance_history[-1])
            #print("\n")
            if distance_history[0] - distance_history[-1] > 0.1:
                additional_reward += 0.1


        '''
        if len(ball_vel_history) >= 5:
            # Calculate if ball velocity is consistently improving
            ball_vel_trend = sum(ball_vel_history[-1] - ball_vel_history[i] for i in range(-5, -1)) / 4
            if ball_vel_trend[0] > 0.01:
                additional_reward += 0.1 # Reward for improving trend toward ball
        '''
        if len(goal_vel_history) >= 5:
            # Calculate if goal velocity is consistently improving
            goal_vel_trend = sum(goal_vel_history[-1] - goal_vel_history[i] for i in range(-5, -1)) / 4
            if goal_vel_trend > 0.05:
                additional_reward += 0.4  # Larger reward for moving ball toward goal
                time_since_touch = 0
            elif goal_vel_trend < -0.05:
                additional_reward -= 0.3  # Penalty for hitting the ball away from goal
                time_since_touch = 0

        reward += additional_reward
        total_reward += reward
        #~~~~~~~~~~~~~calculate reward~~~~~~~~~~~~~#
        #print(reward)

        r = torch.tensor(reward, dtype=torch.float, device=device)
        memory.push((state, action, new_state, r, done))
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        state = new_state
        

        if done:
            total_rewards.append(total_reward)
            plot_avg_reward()
            break

        camera_id = 4
        pixels = env.physics.render(camera_id=camera_id, width=640, height=480)
        if steps_done % 4 == 0:
            image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            cv2.imshow('Soccer', image)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(1)


