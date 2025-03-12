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
from SAC_net import Agent
import csv




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
reward_log_file = "rewards.csv"

# Initialize the soccer environment with random actions
random_state = np.random.RandomState(42)
env = dm_soccer.load(team_size=2,
                     time_limit=10.0,
                     disable_walker_contacts=False,
                     enable_field_box=True,
                     terminate_on_goal=False,
                     walker_type=dm_soccer.WalkerType.BOXHEAD)
env.reset()

# Retrieve action_specs for all players
action_specs = env.action_spec()
timestep = env.reset()

# setup the agents
team1_player1 = Agent(extract_state(timestep.observation[0]).shape[0], action_specs[0].shape[0], action_specs[0].maximum, "player0")
team1_player2 = Agent(extract_state(timestep.observation[0]).shape[0], action_specs[0].shape[0], action_specs[1].maximum, "player1")
team2_player1 = Agent(extract_state(timestep.observation[0]).shape[0], action_specs[0].shape[0], action_specs[2].maximum, "player2")
team2_player2 = Agent(extract_state(timestep.observation[0]).shape[0], action_specs[0].shape[0], action_specs[3].maximum, "player3")

players = [team1_player1, team1_player2, team2_player1, team2_player2]



def episode(capture_video, out=None, frame_width=None, frame_height=None):

    tot_rewards = [0,0,0,0]
    env.reset()

    # Retrieve action_specs for all players
    action_specs = env.action_spec()
    timestep = env.reset()
    states = [] #np.zeros(len(action_specs)) # states of from the perspective of each player

    # Step through the environment with random actions and capture video
    while not timestep.last():  # Capture 100 frames

        print(timestep.last())

        # Take action and observe current state for each player
        actions = []
        # record the current state and next action
        for i in range(len(action_specs)):
            states.append(extract_state(timestep.observation[i]))
            action = players[i].choose_action(states[i])
            actions.append(action)
        
        # Step through the environment with the random actions
        timestep = env.step(actions)

        for i in range(len(action_specs)):
            # record the next state reward and done marker for each player and add to the players buffer
            next_state = extract_state(timestep.observation[i])
            reward = timestep.reward[i]
            done = timestep.last()
            # add MDP tuple to the replay buffer
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

    return tot_returns


# Release the video writer
# ---------------------------------------------------------------------------------------
#out.release()
# ---------------------------------------------------------------------------------------

# Clean up any OpenCV windows
# ---------------------------------------------------------------------------------------
#cv2.destroyAllWindows()
# ---------------------------------------------------------------------------------------

# Print success message
# ---------------------------------------------------------------------------------------
#print(f"Video saved as {video_filename}")
# ---------------------------------------------------------------------------------------



##############
# main loop  #
##############

save_freq = 200 # frequency that the models are saved
video_frequency = 10000 # frequency at which to capture videos of the play
load_model = False
episodes = 0

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
    if episodes % video_frequency == 0:
        record_video = False
        # Video settings
        # ---------------------------------------------------------------------------------------
        video_filename = 'soccer_simulation_video_random_actions' + episodes.__str__() + '.avi'
        frame_width = 640  # Width of the camera view
        frame_height = 480  # Height of the camera view
        # ---------------------------------------------------------------------------------------
        # Create a video writer object (XVID codec)
        # ---------------------------------------------------------------------------------------
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 30  # Frames per second
        out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
        # ---------------------------------------------------------------------------------------
        top_returns = episode(record_video, out=out, frame_width=frame_width, frame_height=frame_height)
    else:
        # normal episode without video
        tot_returns = episode(record_video)

    with open(reward_log_file, "a") as file:
        file.write(f"{tot_returns}\n")

    episodes += 1
    print(episodes, " complete\n")


