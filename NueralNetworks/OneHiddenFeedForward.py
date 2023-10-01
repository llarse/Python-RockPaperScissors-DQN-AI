import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim


class OneHiddenFF(nn.Module):
    def __init__(self, lr, input_dims, hidden_dim_1, hidden_dim_2, n_actions):
        super(OneHiddenFF, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.n_actions = n_actions

        # Functionally define sequental
        self.input_layer = nn.Linear(*self.input_dims, self.hidden_dim_1)
        self.hidden_layer = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.output_layer = nn.Linear(self.hidden_dim_1, self.n_actions)

        # Optimizer and Criterion
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        # Use GPU if available
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = functional.relu(self.input_layer(state))
        x = functional.relu(self.hidden_layer(x))
        actions = self.output_layer(x)
        return actions

    def forwardInterface(self, state):
        call = self.forward(torch.tensor(
            state, dtype=torch.float32).to(self.device))
        return torch.argmax(call).item()
