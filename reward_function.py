import numpy as np

from collections import namedtuple, deque

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RewardFunction:
    
    def __init__(self, queue_length=10, stagnant_timeout=10000, touch_timeout=1000):
        """
        Initialize the Reward Function.
        
        Args:
            queue_length: Length of histories (default = 10)
            stagnant_timeout: Total episode length before timeout (default = 10000)
            touch_timeout: Total time of not touching ball before timeout (default = 1000)
        """
        
        self.ball_vel_history = deque(maxlen=queue_length)  # Store last 10 velocities toward ball
        self.goal_vel_history = deque(maxlen=queue_length)  # Store last 10 velocities of ball toward goal
        self.distance_history = deque(maxlen=queue_length)  

        self.stagnant_timeout = stagnant_timeout
        self.touch_timeout = touch_timeout

        self.stagnant = 0
        self.time_since_touch = 0

        #set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        
    def UpdateAndFetchReward(self, timestep):
        """
        Process the observation and return the reward for the timestep.
        
        Args:
            timestep: The environment timestep containing observations
            
        Returns:
            reward: reward for the current timestep
            next_state: next state in the environment
            scored: bool if ball is scored
            timeout: bool if player hasnt touched ball in long time
        """
        
        self.stagnant+=1
        self.time_since_touch+=1

        scored = False
        timeout = False

        player_pos = timestep.observation[1]['opponent_0_ego_position']
        ball_pos = timestep.observation[1]['ball_ego_position']
        distance_to_ball = np.linalg.norm(player_pos[0] - ball_pos[0])

        #ball_vel is velocity to the ball
        #goal_vel is ball's velocity to goal
        next_state, ball_vel, goal_vel = self._get_observation_tensor(timestep, player_idx=0, device=self.device)
        self.ball_vel_history.append(ball_vel)
        self.goal_vel_history.append(goal_vel)
        self.distance_history.append(distance_to_ball)

        reward = timestep.reward
        reward = reward[0]*10


        #~~~~~~~~~~~~~calculate reward~~~~~~~~~~~~~#
        #if the reward is > 0, that means a goal has been scored and we reset the environment
        if reward > 0:
            scored = True
            reward+=300

        #if it hasn't scored a goal within the time limit we just reset
        if self.stagnant > self.stagnant_timeout:
            timeout = True
            reward-=50

        #if it hasn't touched the ball within a certain time limit, we reset earlier
        if self.time_since_touch > self.touch_timeout:
            timeout = True
            reward-=50

        #tack on additions rewards beside the goal
        additional_reward = 0

        if len(self.distance_history) >= 5:
            #print(distance_history[0])
            #print(distance_history[-1])
            #print("\n")
            if self.distance_history[0] - self.distance_history[-1] > 0.1:
                additional_reward += 0.1

        if len(self.goal_vel_history) >= 5:
            # Calculate if goal velocity is consistently improving
            goal_vel_trend = sum(self.goal_vel_history[-1] - self.goal_vel_history[i] for i in range(-5, -1)) / 4
            if goal_vel_trend > 0.05:
                additional_reward += 0.4  # Larger reward for moving ball toward goal
                self.time_since_touch = 0
            elif goal_vel_trend < -0.05:
                additional_reward -= 0.3  # Penalty for hitting the ball away from goal
                self.time_since_touch = 0

        reward += additional_reward

        return reward, next_state, scored, timeout
    
    def reset(self):
        """
        Resets the reward function. Execute at the start of every episode.
        """

        self.ball_vel_history.clear()
        self.goal_vel_history.clear()
        self.distance_history.clear()
        self.stagnant = 0
        self.time_since_touch = 0

    def _process_observation(self, timestep, player_idx=0):
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
    
    def _get_observation_tensor(self, timestep, player_idx=0, device='cuda'):
        """
        Convert observation to a tensor ready for the neural network.
        """
        flat_obs, ball_vel, goal_vel = self._process_observation(timestep, player_idx)
        
        # Convert to tensor and add batch dimension
        obs_tensor = torch.tensor(flat_obs, dtype=torch.float32).unsqueeze(0).to(device)
        return obs_tensor, ball_vel, goal_vel
    