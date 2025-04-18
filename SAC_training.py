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
from reward_function import RewardFunction


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
    plt.ylim([-20,300])
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
        np.array(obs['teammate_0_ego_position']).flatten(),
        np.array(obs['teammate_0_ego_linear_velocity']).flatten(),
        np.array(obs['teammate_0_ego_orientation']).flatten(),
        np.array(obs['opponent_0_ego_position']).flatten(),
        np.array(obs['opponent_0_ego_linear_velocity']).flatten(),
        np.array(obs['opponent_0_ego_orientation']).flatten(),
        np.array(obs['opponent_1_ego_position']).flatten(),
        np.array(obs['opponent_1_ego_linear_velocity']).flatten(),
        np.array(obs['opponent_1_ego_orientation']).flatten(),
        np.array([obs['stats_vel_to_ball']]).flatten(),  # Single value stats
        np.array([obs['stats_vel_ball_to_goal']]).flatten(),
        np.array([obs['stats_home_avg_teammate_dist']]).flatten(),
        np.array([obs['stats_home_score']]).flatten(),
        np.array([obs['stats_away_score']]).flatten()
    ])
    
    return state_vector

# file name that will hold the rewards as episodes continue
reward_log_file = "rewards_2.csv"

# Initialize the soccer environment with random actions
random_state = np.random.RandomState(42)
env = dm_soccer.load(team_size=2,
                     time_limit=50000.0,
                     disable_walker_contacts=False,
                     enable_field_box=True,
                     terminate_on_goal=True,
                     walker_type=dm_soccer.WalkerType.BOXHEAD)
env.reset()

# Retrieve action_specs for all players
action_specs = env.action_spec()
timestep = env.reset()

# setup the agents
team1_player1 = Agent(extract_state(timestep.observation[0]).shape[0], action_specs[0].shape[0], action_specs[0].maximum, "player0", tau=0.01, reward_scale=4)
team1_player2 = Agent(extract_state(timestep.observation[0]).shape[0], action_specs[0].shape[0], action_specs[1].maximum, "player1", tau=0.01, reward_scale=4)
team2_player1 = Agent(extract_state(timestep.observation[0]).shape[0], action_specs[0].shape[0], action_specs[0].maximum, "player4", tau=0.01, reward_scale=4)
team2_player2 = Agent(extract_state(timestep.observation[0]).shape[0], action_specs[0].shape[0], action_specs[0].maximum, "player5", tau=0.01, reward_scale=4)


players = [team1_player1, team1_player2, team2_player1, team2_player2]



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
            if i == 0 or i == 1:
                action = players[i].choose_action(states[i])
                actions.append(action)
            else:
                action = np.random.uniform(action_specs[i].minimum, action_specs[i].maximum, size=action_specs[i].shape)
                actions.append(action)
        
        # Step through the environment with the random actions
        timestep = env.step(actions)

        for i in range(len(action_specs)):
            # record the next state reward and done marker for each player and add to the players buffer
            next_state = extract_state(timestep.observation[i])
            reward, _, scored, timeout = reward_functions[i].UpdateAndFetchReward(timestep, i)
            done = timestep.last() or scored or timeout
            # add MDP tuple to the replay buffer
            if i == 0 or i == 1:
                players[i].remember(states[i], actions[i], reward, next_state, done)
                players[i].learn()
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
video_frequency = 500 # frequency at which to capture videos of the play
load_model = False
episodes = 0
tot_returns_tot = []
reward_function1 = RewardFunction()
reward_function2 = RewardFunction()
reward_function3 = RewardFunction()
reward_function4 = RewardFunction()
reward_functions = [reward_function1, reward_function2, reward_function3, reward_function4]

if load_model:
    for i in range(len(action_specs)):
        players[i].load_models()

while True:
    env.reset()
    record_video = False
    tot_returns = [0,0,0,0]
    print("Episode {} start".format(episodes))
    # save the models every so many episodes
    if episodes % save_freq == 0:
        for i in range(len(action_specs)):
            players[i].save_models()

    # record the run every so many episodes
    if episodes % video_frequency == 1:
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
    print(episodes, " complete\n")
    tot_returns_tot.append(tot_returns)
    plot_avg_reward()


