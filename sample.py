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

# Set up flags (for any optional arguments)
FLAGS = flags.FLAGS
flags.DEFINE_enum("walker_type", "BOXHEAD", ["BOXHEAD", "ANT", "HUMANOID"], "The type of walker to explore with.")
flags.DEFINE_bool("enable_field_box", True, "Enable the physical bounding box enclosing the ball.")
flags.DEFINE_bool("disable_walker_contacts", False, "Disable walker-walker contacts.")
flags.DEFINE_bool("terminate_on_goal", False, "Terminate on goal.")

# Initialize the soccer environment with random actions
random_state = np.random.RandomState(42)
env = dm_soccer.load(team_size=2,
                     time_limit=10.0,
                     disable_walker_contacts=False,
                     enable_field_box=True,
                     terminate_on_goal=False,
                     walker_type=dm_soccer.WalkerType.BOXHEAD)
env.reset()

# Video settings
# ---------------------------------------------------------------------------------------
#video_filename = 'soccer_simulation_video_random_actions.avi'
#frame_width = 640  # Width of the camera view
#frame_height = 480  # Height of the camera view
# ---------------------------------------------------------------------------------------

# Create a video writer object (XVID codec)
# ---------------------------------------------------------------------------------------
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fps = 30  # Frames per second
#out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
# ---------------------------------------------------------------------------------------

# Retrieve action_specs for all players
action_specs = env.action_spec()
timestep = env.reset()

# Step through the environment with random actions and capture video
while not timestep.last():  # Capture 100 frames

    # Take random actions for each player
    actions = []
    for action_spec in action_specs:
        action = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)
        actions.append(action)
    
    # Step through the environment with the random actions
    timestep = env.step(actions)

    for i in range(len(action_specs)):
      # prints obesrvation of each player for each step
      print(
        "observations = \n\n\n\n{} {}.".format(timestep.observation[i]['stats_vel_to_ball'],i ))

    # Capture a single camera view
    # ---------------------------------------------------------------------------------------
    #camera_id = 0  # Use the first camera
    #pixels = env.physics.render(camera_id=camera_id, width=frame_width, height=frame_height)
    # ---------------------------------------------------------------------------------------

    # Convert the rendered image to BGR format for OpenCV
    # ---------------------------------------------------------------------------------------
    #image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    # ---------------------------------------------------------------------------------------

    # Write the image to the video file
    # ---------------------------------------------------------------------------------------
    #out.write(image)
    # ---------------------------------------------------------------------------------------


    # Optionally display the current frame
    # ---------------------------------------------------------------------------------------
    #cv2.imshow('Soccer Simulation', image)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break
    # ---------------------------------------------------------------------------------------

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
