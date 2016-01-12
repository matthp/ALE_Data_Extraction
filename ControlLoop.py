import sys
from random import randrange
from ale_python_interface import ALEInterface
import numpy as np
import png
from matplotlib import pyplot as plt
from ReactiveFQI import FittedQController, QIterator, encodeFrame

# Define the path to the directory where the games files are stored
ROMPath = '/Users/admin/Desktop/ROMS/'

# Define the path to the directory where screenshots should be saved
savePath = '/Users/admin/Desktop/ALE ScreenShots/'

# Instantiate the arcade learning environment
ale = ALEInterface()
ale.setInt('random_seed', 5)

# Load the game into the ALE
romFile = ROMPath + 'breakout.bin'
ale.loadROM(romFile)

# Get the list of legal actions
legal_actions = ale.getLegalActionSet()

# Initialize controller
controller = FittedQController(numActions=len(legal_actions), numFeatures=5000, horizon=100, discountParameter=0.3, epsilon=0.2)

# Simulate episodes
numEpisodes = 100

counter = 1  # Total number of time steps, used for naming screenshots with the time index they were taken

for episode in xrange(numEpisodes):

    # Initialize reward to zero as we do not have any reward yet
    reward = 0
    total_reward = 0

    while not ale.game_over():
        # Retrieve and encode current frame
        frame = ale.getScreenRGB()
        frame = np.array(frame, dtype=float)
        frameCode = encodeFrame(frame)

        # Get action from controller, and pass in frame code and previous reward (frame code is dct of flattened frame,
        # can be replaced with deep autoencoder, etc.)
        a = legal_actions[controller.queryForActionAndUpdateExperience(features=frameCode, reward=reward)]

        # Save screen shot of game
        #image = png.from_array(frame, 'RGB;8')
        #imFile = savePath + str(counter) + '.png'
        #image.save(imFile)
        #counter += 1
        #image = []

        # Select an action and send to the ALE. Get reward and add to running tally
        reward = ale.act(a)
        total_reward += reward

    print total_reward

    # Use accumulated experience in the controller to compute the new policy using FQI algorithm and sklearn random
    # forests
    controller.updatePolicyUsingExperience()

    # Reset the ALE for the next episodes
    ale.reset_game()

