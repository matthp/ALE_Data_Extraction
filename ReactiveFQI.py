from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy.fftpack import dct
import random
from random import randrange

# Returns a one dimensional vector encoding a single frame from the ALE
# Tested
def encodeFrame(frame):
    # Number of DCT coefficients to include in the code
    codeLength = 5000

    # Convert frame to numpy array
    frame = np.array(frame, dtype=float)

    # Calculate the total number of elements in the frame
    totalElements = np.product(frame.shape)

    # Reshape the frame to a single dimension
    frame = frame.reshape(1, totalElements)

    # Compute the dct coefficients of the frame and take the first codeLength elements as the frame code
    frameCode = dct(frame)
    frameCode = frameCode[:, 0:codeLength]

    return frameCode


# Uses a base regression algorithm to implement the fitted Q iteration algorithm for a finite number of iterations
# in order to compute a Q estimate from a batch of stored experience
class QIterator:
    # Constructor
    def __init__(self, numActions, discount, horizon):
        self.Models = []
        for action in range(numActions):
            self.Models.append(RandomForestRegressor(n_estimators=12, max_depth=None, max_features="sqrt", min_samples_split=60, min_samples_leaf=30, n_jobs=6))

        self.NumActions = numActions
        self.Discount = discount
        self.Horizon = horizon
        self.Trained = False

    def isTrained(self):
        return self.Trained

    # Iterates the Q function based on observed action dependant transitions and rewards
    def iterateToHorizon(self, S, A, R, S_prime):
        self.Trained = True

        # Get the number of training experience examples
        numExamples = len(A)

        # Assign the initial targets as the one step reward
        targets = []
        for n in range(self.NumActions):
            targets.append(np.zeros((numExamples, 1)))

        for n in range(numExamples):
            action = A[n]
            targets[action][n] = R[n]

        # Train first Q function
        self.iterateRegressor(S, targets)

        # Repeat iteration until Q function extends to horizon
        for h in range(0, self.Horizon):
            # Compute targets for next Q function
            for n in range(0, numExamples):
                action = A[n]
                target[n, action] = R[n] + self.Discount*max(self.predict(S_prime[n, ...]))

            # Train next Q function
            self.iterateRegressor(S, target)

    # Takes action dependant targets and features and fits a regressor
    def iterateRegressor(self, S, target):
        self.clearRegressor()
        self.Model.fit(S, target)

    # Returns the regressor predicted expected value for each action
    def predict(self, S):
        return self.Model.predict(S)

    # Replaces the regressor with a fresh random forest, used to iterate over Q functions
    def clearRegressor(self):
        self.Model = RandomForestRegressor(n_estimators=12, max_depth=None, max_features="sqrt", min_samples_split=60, min_samples_leaf=30, n_jobs=6)


class FittedQController:

    # Constructor
    def __init__(self, numActions, numFeatures, horizon, discountParameter, epsilon):
        self.NumActions = numActions    # The number of possible discrete actions
        self.NumFeatures = numFeatures  # The number of features in the input space
        self.Horizon = horizon  # The horizon for which we care about accurate Q estimates
        self.DiscountParameter = discountParameter  # The discount factor applied to future rewards
        self.Epsilon = epsilon  # The probability of selecting a random action during the exploration phase

        self.QFunction = QIterator(numActions=numActions, discount=discountParameter, horizon=horizon)   # Learned Q* function used for greedy policy

        self.PriorAction = 0    # The action executed on the previous time step, initially the first legal action
        self.PriorS = np.zeros((1, numFeatures))  # The state in the previous time step, all zeros at start of episode

        # Stores experience for Q function training
        self.S = np.zeros((1, numFeatures))
        self.A = np.zeros(1)
        self.R = np.zeros(1)
        self.S_prime = np.zeros((1, numFeatures))

    def queryForActionAndUpdateExperience(self, features, reward):

        action = 0

        if self.QFunction.isTrained():
            if random.uniform(0,1) > self.Epsilon:
                # Compute all Q values
                qValues = self.QFunction.predict(features)

                # Select greedy action as action with maximum Q value
                action = qValues.index(max(qValues))
            else:
                action = randrange(self.NumActions)
        else:
            action = randrange(self.NumActions)

        # Store experience
        self.A = np.concatenate([self.A, [self.PriorAction]])
        self.R = np.concatenate([self.R, [reward]])
        self.S = np.concatenate([self.S, self.PriorS], axis=0)
        self.S_prime = np.concatenate([self.S_prime, features], axis=0)

        # Update prior information
        self.PriorAction = action
        self.PriorS = features

        return action

    # Used for purely greedy control, when the policy is no longer being updated
    def queryForAction(self, features):

        action = 0

        if self.QFunction.isTrained():
            # Compute all Q values
            qValues = self.QFunction.predict(features)

            # Select greedy action as action with maximum Q value
            action = qValues.index(max(qValues))

        # Update prior information
        self.PriorAction = action
        self.PriorS = features

        return action

    # Runs the Q function through a round of FQI to generate a better greedy policy based on newly aquired experience
    def updatePolicyUsingExperience(self):
        # Check to make sure enough experience has been aquired for an initial Q function estimate,
        # if so then pass experience to Q iterator and train a Q function, otherwise return

        requiredExperience = 500

        if len(self.A) >= requiredExperience:
            self.QFunction.iterateToHorizon(S=self.S, A=self.A, R=self.R, S_prime=self.S_prime)


