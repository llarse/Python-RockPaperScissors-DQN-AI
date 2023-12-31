import torch
import numpy as np


class DQNAgent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.05, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        # initialize memories with zeros
        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, next_state, terminal):
        ''' Stores the action, the consequence of that action,
            and the environement before and after the action'''
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        # Epsilon-greedy
        if np.random.random() > self.epsilon:
            # Exploit - take action based on a forward pass to the network
            state = torch.tensor(np.array(observation),
                                 dtype=torch.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            # Explore - take action based on a random choice
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        # reset the optimizer gradients
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        # Create a batch
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # Send batch variables to device
        state_batch = torch.tensor(
            self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(
            self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = torch.tensor(
            self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(
            self.terminal_memory[batch]).to(self.Q_eval.device)

        # calculate q values
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)[
            batch_index, action_batch]
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * \
            torch.max(q_next.view(-1, 1), dim=1)[0]

        # calculate loss
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min
        return loss
