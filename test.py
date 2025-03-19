# ===========================================================================================================
# Sample.py is a modification to the random action simulation of boxhead players
# The parts of code that are surrounded by "# ------..." are for capturing video (currently commented out)
# If a screen with the field pops up when the code is running, then close the pop up as it freezes the code
# The code runs faster with no video recording and no print statements
# ===========================================================================================================
import numpy as np
import cv2
import functools
from dm_control.locomotion import soccer as dm_soccer
from absl import app, flags
import PIL.Image
from SAC_pythonFiles.SAC_net import Agent 
import csv
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
import torch


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


def plot_avg_reward(show_result=False):
    plt.figure(2)
    rewards_t = torch.tensor(tot_returns_tot, dtype=torch.float)
    if show_result:
        plt.title('Reward')
    else:
        plt.clf()
        plt.title('Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    # if len(rewards_t) >= 100:
    #     means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


player_vel_history = [deque(maxlen=10),deque(maxlen=10)]  # Store last 10 velocities toward ball
ball_vel_history = [deque(maxlen=10),deque(maxlen=10)]  # Store last 10 velocities of ball toward goal
stagnant = 0
time_since_touch = 0
back = 0
forw = 0

# function to get the rewards for moving toward the ball and moving ball to the goal
def get_reward(player_vel, ball_vel, time_since_touch, player):
    global forw
    global back
    player_vel_history[player].append(player_vel)
    ball_vel_history[player].append(ball_vel)
    

    additional_reward = 0
    if len(ball_vel_history[player]) >= 5:
        # Calculate if ball velocity is consistently improving
        ball_vel_trend = sum(ball_vel_history[player][-1] - ball_vel_history[player][i] for i in range(-5, -1)) / 4
        player_vel_trend = sum(player_vel_history[player][-1] - player_vel_history[player][i] for i in range(-5, -1)) / 4
        if player_vel_trend > 0.01:
            additional_reward += 0.1  # Reward for improving trend toward ball
        if player_vel < -0.01 and ball_vel_trend < 0.01:
            additional_reward -= 0.3
        if player_vel > 0.01:
            additional_reward += 0.6*player_vel[0]

    if len(ball_vel_history[player]) >= 5:
        # Calculate if goal velocity is consistently improving
        ball_vel_trend = sum(ball_vel_history[player][-1] - ball_vel_history[player][i] for i in range(-5, -1)) / 4
        if ball_vel > 0.05:
            additional_reward += 1.  # Larger reward for moving ball toward goal
            time_since_touch = 0
        elif ball_vel < -0.05:
            additional_reward -= 0.8  # Penalty for hitting the ball away from goal
            time_since_touch = 0
    if time_since_touch == 2000:
        additional_reward -= 200
    return additional_reward



def extract_state(obs):
    state_vector = np.concatenate([
        np.array(obs['body_height']).flatten(),
        np.array(obs['joints_pos']).flatten(),
        np.array(obs['joints_vel']).flatten(),
        np.array(obs['sensors_accelerometer']).flatten(),
        np.array(obs['sensors_gyro']).flatten(),
        np.array(obs['sensors_velocimeter']).flatten(),
        np.array(obs['prev_action']).flatten(),
        np.array(obs['ball_ego_position']).flatten(),
        np.array(obs['ball_ego_linear_velocity']).flatten(),
        np.array(obs['ball_ego_angular_velocity']).flatten(),
        #np.array(obs['teammate_0_ego_position']).flatten(),
        #np.array(obs['teammate_0_ego_linear_velocity']).flatten(),
        #np.array(obs['teammate_0_ego_orientation']).flatten(),
        np.array(obs['opponent_0_ego_position']).flatten(),
        np.array(obs['opponent_0_ego_linear_velocity']).flatten(),
        np.array(obs['opponent_0_ego_orientation']).flatten(),
        #np.array(obs['opponent_1_ego_position']).flatten(),
        #np.array(obs['opponent_1_ego_linear_velocity']).flatten(),
        #np.array(obs['opponent_1_ego_orientation']).flatten(),
        np.array([obs['stats_vel_to_ball']]).flatten(),  # Single value stats
        np.array([obs['stats_vel_ball_to_goal']]).flatten(),
        np.array([obs['stats_home_avg_teammate_dist']]).flatten(),
        np.array([obs['stats_home_score']]).flatten(),
        np.array([obs['stats_away_score']]).flatten()
    ])
    
    return state_vector

# file name that will hold the rewards as episodes continue
reward_log_file = "rewards.csv"

# Initialize the soccer environment with random actions
random_state = np.random.RandomState(42)
env = dm_soccer.load(team_size=1,
                     time_limit=300.0,
                     disable_walker_contacts=False,
                     enable_field_box=True,
                     terminate_on_goal=True,
                     walker_type=dm_soccer.WalkerType.BOXHEAD)
env.reset()

# Retrieve action_specs for all players
action_specs = env.action_spec()
timestep = env.reset()

# setup the agents
team1_player1 = Agent(extract_state(timestep.observation[0]).shape[0], action_specs[0].shape[0], action_specs[0].maximum, "player", tau=0.01)
team2_player1 = 0


players = [team1_player1, team2_player1]



def episode(capture_video, out=None, frame_width=None, frame_height=None):

    tot_rewards = [0,0,0,0]
    env.reset()
    stagnant = 0
    time_since_touch = 0

    # Retrieve action_specs for all players
    action_specs = env.action_spec()
    timestep = env.reset()
    states = [] #np.zeros(len(action_specs)) # states of from the perspective of each player

    # Step through the environment with random actions and capture video
    while not timestep.last() and time_since_touch < 2000:  # Capture 100 frames

        stagnant+=1
        time_since_touch+=1

        # Take action and observe current state for each player
        actions = []
        states = []
        # record the current state and next action
        for i in range(len(action_specs)):
            states.append(extract_state(timestep.observation[i]))
            if i == 0:
                action = team1_player1.choose_action(states[i])
                actions.append(action)
            else:
                action = np.random.uniform(action_specs[i].minimum, action_specs[i].maximum, size=action_specs[i].shape)
                actions.append(action)
        
        # Step through the environment with the random actions
        timestep = env.step(actions)

        for i in range(len(action_specs)):
            # record the next state reward and done marker for each player and add to the players buffer
            next_state = extract_state(timestep.observation[i])
            reward = (500*timestep.reward[i]) + get_reward(timestep.observation[i]['stats_vel_to_ball'],timestep.observation[i]['stats_vel_ball_to_goal'], time_since_touch, i)
            done = timestep.last() or time_since_touch == 2000
            # add MDP tuple to the replay buffer
            if i ==0:
                team1_player1.remember(states[i], actions[i], reward, next_state, done)
                team1_player1.learn()
            tot_rewards[i] += reward

        if capture_video:
            # Capture a single camera view
            # ---------------------------------------------------------------------------------------
            camera_id = 0  # Use the first camera
            pixels = env.physics.render(camera_id=camera_id, width=frame_width, height=frame_height)
            # ---------------------------------------------------------------------------------------

            # Convert the rendered image to BGR format for OpenCV
            # ---------------------------------------------------------------------------------------
            image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            # ---------------------------------------------------------------------------------------

            if stagnant % 4 == 0:
                image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                cv2.imshow('Soccer', image)

            # Write the image to the video file
            # ---------------------------------------------------------------------------------------
            out.write(image)
            # ---------------------------------------------------------------------------------------

    print(time_since_touch)

    return tot_rewards


##############
# main loop  #
##############

save_freq = 5 # frequency that the models are saved
video_frequency = 200 # frequency at which to capture videos of the play
load_model = True
episodes = 0
tot_returns_tot = []

if load_model:
    for i in range(len(action_specs)):
        team1_player1.load_models()

while True:
    env.reset()
    record_video = False
    tot_returns = [0,0,0,0]
    print("Episode {} start".format(episodes))
    # save the models every so many episodes
    if episodes % save_freq == 0:
        team1_player1.save_models()

    # record the run every so many episodes
    if episodes % video_frequency == 0:
        record_video = True
        # Video settings
        # ---------------------------------------------------------------------------------------
        video_filename = 'soccer_simulation_video_random_actions'+ episodes.__str__()+'.avi'
        frame_width = 640  # Width of the camera view
        frame_height = 480  # Height of the camera view
        # ---------------------------------------------------------------------------------------
        # Create a video writer object (XVID codec)
        # ---------------------------------------------------------------------------------------
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 30  # Frames per second
        out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
        # ---------------------------------------------------------------------------------------

        tot_returns = episode(record_video, out=out, frame_width=frame_width, frame_height=frame_height)

        # Release the video writer
        # ---------------------------------------------------------------------------------------
        out.release()
        # ---------------------------------------------------------------------------------------

        # Clean up any OpenCV windows
        # ---------------------------------------------------------------------------------------
        cv2.destroyAllWindows()
        # ---------------------------------------------------------------------------------------

        # Print success message
        # ---------------------------------------------------------------------------------------
        print(f"Video saved as {video_filename}")
        # ---------------------------------------------------------------------------------------

    else:
        # normal episode without video
        tot_returns = episode(record_video)

    with open(reward_log_file, "a") as file:
        file.write(f"{tot_returns}\n")

    episodes += 1
    forw = 0
    back = 0
    print(episodes, " complete\n")
    tot_returns_tot.append(tot_returns)
    plot_avg_reward()


