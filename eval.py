from tqdm import tqdm
import random
import numpy as np

from Agents.DQNAgent import DQNAgent
from NueralNetworks.OneHiddenFeedForward import OneHiddenFF
from Plots.RockPaperScissorsTracker import RPSTracker
from Plots.PlotWinRates import plotWinRate
# ----------------------------------------------------------------------------------------------- Module Globals
# Plot globals
SAVE_FILE = 'RockPaperScissorsEval.PNG'
CATEGORIES = ['Wins', 'Ties', 'Losses']
INITIAL_HEIGHTS = [0, 0, 0]
X_LABEL = 'Progress'
Y_LABEL = 'Frequency'
TITLE = 'Games: 0, Episode: 0'

# Game globals
EPISODES = 100
GAMES_PER_EPISODE = 100


# ---------------------------------------------------------------------------------------------------- Hyperparamaters
GAMMA = 0.99
EPSILON = 1.0
BATCH_SIZE = 64
EPS_DEC = 5e-4
EPS_END = 0.001
INPUT_DIMS = 3
LR = 0.001

HIDDEN_DIMS = [256, 256]

# ---------------------------------------------------------------------------------------------------- Classes/Functions


def one_hot_encode(move):
    ''' one hot encodes move
    args: move = ROCK, PAPER, or SCISSORS'
    retuns: one_hot = [1,0,0] where the 1 is in the place of the move'''
    one_hot = np.zeros(3)
    one_hot[move] = 1
    return one_hot


class OneHiddenDQNAgent(DQNAgent):
    ''' Give the DQNAgent a network to train'''

    def __init__(self, hidden_dims, *args, **kwargs):
        super(OneHiddenDQNAgent, self).__init__(*args, **kwargs)
        self.hidden_dims = hidden_dims
        self.Q_eval = OneHiddenFF(
            self.lr, self.input_dims, *hidden_dims, self.n_actions)


# -------------------------------------------------------------------------------------------------- main
if __name__ == "__main__":
    tracker = RPSTracker(CATEGORIES, INITIAL_HEIGHTS, X_LABEL, Y_LABEL, TITLE)
    # Initialize player DQN agent in "train" mode
    random.seed(31415926)  # For Repeatability
    player = OneHiddenDQNAgent(HIDDEN_DIMS, gamma=GAMMA, epsilon=EPSILON, batch_size=BATCH_SIZE,
                               n_actions=3, eps_dec=EPS_DEC, eps_end=EPS_END, input_dims=[INPUT_DIMS], lr=LR)
    # Init Opponent with arbitrary learn rate
    opponent = OneHiddenFF(lr=0.001, input_dims=[
                           INPUT_DIMS], hidden_dim_1=HIDDEN_DIMS[0], hidden_dim_2=HIDDEN_DIMS[1], n_actions=3)
    # set oppenent state
    opponent_state = one_hot_encode(random.randint(0, 2))
    # get opponent action
    opponent_action = opponent.forwardInterface(opponent_state)
    # get current state
    state = one_hot_encode(opponent_action)
    for episode in tqdm(range(EPISODES)):
        done = False
        while not done:
            # get player action
            player_action = player.choose_action(state)
            # encode opponents state
            opponent_state = one_hot_encode(player_action)
            # get opponent action
            opponent_action = opponent.forwardInterface(opponent_state)
            # get reward
            reward = tracker.decide_winner(player_action, opponent_action)
            # update "environment"
            tracker.draw()
            # get next state
            next_state = one_hot_encode(opponent_action)
            # Termination Conditions
            if tracker.games == GAMES_PER_EPISODE:
                tracker.reset()
                done = True
            # learn
            player.store_transition(
                state, player_action, reward, next_state, done)
            loss = player.learn()
            # set state to next state
            state = next_state
        # Uncomment to print score and epsilon - Will mess up TQDM slightly
        '''print('Episode ', episode, 'score %.2f' % reward,
              'epsilon %.4f' % player.epsilon)'''
    rounds = [i+1 for i in range(EPISODES)]
    plotWinRate(rounds, tracker.win_rates, rounds, SAVE_FILE)
