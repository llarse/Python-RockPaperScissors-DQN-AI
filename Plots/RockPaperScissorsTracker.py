from Plots.BarPlot import BarPlot

ROCK = 0
SCISSORS = 1
PAPER = 2

WIN_REWARD = 1
LOSE_PUNISHMENT = -1
TIE_PUNISHMENT = -0.25


class RPSTracker(BarPlot):
    def __init__(self, *args, **kwargs):
        super(RPSTracker, self).__init__(*args, **kwargs)
        # Score formated as Wins, ties, losses
        self.score = [0, 0, 0]
        self.games = 0

        self.win_rates = []
        self.episode = 0

    def decide_winner(self, player_action, oppenent_action):
        self.games += 1
        reward = 0

        # Rock, Paper, Scissors logic
        # Win
        if (
            (player_action == ROCK and oppenent_action == SCISSORS)
            or (player_action == SCISSORS and oppenent_action == PAPER)
            or (player_action == PAPER and oppenent_action == ROCK)
        ):
            reward = WIN_REWARD
            self.score[0] += 1
        # Tie
        elif (player_action == oppenent_action):
            reward = TIE_PUNISHMENT
            self.score[1] += 1
        # Lose
        else:
            reward = LOSE_PUNISHMENT
            self.score[2] += 1

        # update visual data
        self.heights = self.score
        self.title = f"Games: {self.games}, Episodes: {self.episode + 1}"
        return reward

    def reset(self):
        self.episode += 1
        self.win_rates.append((self.score[0] / self.games))

        self.games = 0
        self.score = [0, 0, 0]
