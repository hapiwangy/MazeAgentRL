import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class A2CNetwork(nn.Module):
    def __init__(self, obs_shape=(3, 3), num_actions=4, hidden_size=64):
        super(A2CNetwork, self).__init__()

        self.hidden_size = hidden_size

        # 1. Observation encoder
        # The local view is a 3x3 integer grid (0 to 5), embedded into continuous vectors.
        num_classes = 6  # 0: path, 1: wall, 2: start, 3: exit, 4: key, 5: agent
        embedding_dim = 8
        self.embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=embedding_dim)

        # Flattened dimension: 3 * 3 * 8 = 72
        flattened_dim = obs_shape[0] * obs_shape[1] * embedding_dim

        # 2. Recurrent layer
        # Stores trajectory history to handle partial observability.
        self.gru = nn.GRU(input_size=flattened_dim, hidden_size=hidden_size, batch_first=True)

        # 3. Actor network
        # Outputs action logits for 4 actions.
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
        )

        # 4. Critic network
        # Outputs a scalar value estimate for the current state.
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, hidden_state):
        """
        x: (batch_size, sequence_length, 3, 3)
        hidden_state: (1, batch_size, hidden_size)
        """
        batch_size, seq_len, h, w = x.shape

        # (batch_size, seq_len, 3, 3) -> (batch_size, seq_len, 3, 3, embedding_dim)
        x = self.embedding(x.long())

        # Flatten to (batch_size, seq_len, 72)
        x = x.view(batch_size, seq_len, -1)

        # Pass through the GRU layer.
        out, new_hidden_state = self.gru(x, hidden_state)

        # Use only the output from the last time step.
        out = out[:, -1, :]

        # Compute actor and critic outputs.
        action_logits = self.actor(out)
        state_value = self.critic(out)

        return action_logits, state_value, new_hidden_state

    def init_hidden(self, batch_size=1):
        """Initialize the GRU hidden state at the start of each episode."""
        return torch.zeros(1, batch_size, self.hidden_size)


class A2CAgent:
    def __init__(self, network):
        self.network = network
        self.hidden_state = None

    def reset_memory(self):
        """Reset recurrent memory at the start of each episode."""
        self.hidden_state = self.network.init_hidden()

    def select_action(self, obs):
        """
        Select an action from the current observation and return the values needed for loss computation.
        obs shape: (3, 3) numpy array
        """
        # Convert the numpy array to a tensor and add batch and sequence dimensions: (1, 1, 3, 3)
        obs_tensor = torch.tensor(obs, dtype=torch.long).unsqueeze(0).unsqueeze(0)

        # Forward pass through the network.
        logits, state_value, self.hidden_state = self.network(obs_tensor, self.hidden_state)

        # Sample an action from the categorical distribution.
        probs = F.softmax(logits, dim=-1)
        distribution = Categorical(probs)
        action = distribution.sample()

        # Save the log probability for the actor loss.
        log_prob = distribution.log_prob(action)

        # Return the action, log probability, state value, and entropy bonus term.
        return (
            action.item(),
            log_prob.unsqueeze(0),
            state_value.view(1),
            distribution.entropy().unsqueeze(0),
        )
