import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class REINFORCENetwork(nn.Module):
    def __init__(self, obs_shape=(3, 3), num_actions=4, hidden_size=64):
        super(REINFORCENetwork, self).__init__()

        self.hidden_size = hidden_size

        # 1. Observation encoder, aligned with the A2C representation.
        num_classes = 6
        embedding_dim = 8
        self.embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=embedding_dim)
        flattened_dim = obs_shape[0] * obs_shape[1] * embedding_dim

        # 2. Recurrent memory for partially observable maze navigation.
        self.gru = nn.GRU(input_size=flattened_dim, hidden_size=hidden_size, batch_first=True)

        # 3. Policy head. REINFORCE does not use a critic branch.
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
        )

    def forward(self, x, hidden_state):
        batch_size, seq_len, h, w = x.shape
        x = self.embedding(x.long())
        x = x.view(batch_size, seq_len, -1)

        out, new_hidden_state = self.gru(x, hidden_state)
        out = out[:, -1, :]

        action_logits = self.actor(out)
        return action_logits, new_hidden_state

    def init_hidden(self, batch_size=1):
        """Initialize the recurrent hidden state for a new episode."""
        return torch.zeros(1, batch_size, self.hidden_size)


class REINFORCEAgent:
    def __init__(self, network):
        self.network = network
        self.hidden_state = None

    def reset_memory(self):
        """Reset recurrent memory at the start of each episode."""
        self.hidden_state = self.network.init_hidden()

    def select_action(self, obs):
        """Sample an action and return the policy statistics required for training."""
        obs_tensor = torch.tensor(obs, dtype=torch.long).unsqueeze(0).unsqueeze(0)

        logits, self.hidden_state = self.network(obs_tensor, self.hidden_state)

        probs = F.softmax(logits, dim=-1)
        distribution = Categorical(probs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        # REINFORCE stores only policy-related terms.
        return (
            action.item(),
            log_prob.unsqueeze(0),
            distribution.entropy().unsqueeze(0),
        )
