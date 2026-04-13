import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class REINFORCENetwork(nn.Module):
    def __init__(self, obs_shape=(3, 3), num_actions=4, hidden_size=64):
        super(REINFORCENetwork, self).__init__()

        self.hidden_size = hidden_size

        # 1. Observation encoder
        num_classes = 6
        embedding_dim = 8
        self.embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=embedding_dim)
        flattened_dim = obs_shape[0] * obs_shape[1] * embedding_dim

        # 2. Recurrent memory
        # Adding +1 to input_size to accommodate the 'has_key' flag
        self.gru = nn.GRU(input_size=flattened_dim + 1, hidden_size=hidden_size, batch_first=True)

        # 3. Policy head
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
        )

    def forward(self, x, has_key, hidden_state):
        """
        x: (batch_size, seq_len, 3, 3)
        has_key: (batch_size, seq_len, 1) float tensor
        hidden_state: (1, batch_size, hidden_size)
        """
        batch_size, seq_len, h, w = x.shape
        
        # Embed and flatten the grid view
        x = self.embedding(x.long())
        x = x.view(batch_size, seq_len, -1)

        # Concatenate the 'has_key' flag to the flattened visual features
        # This allows the policy to branch logic based on inventory state
        combined_features = torch.cat((x, has_key), dim=-1) # (batch, seq, 73)

        out, new_hidden_state = self.gru(combined_features, hidden_state)
        out = out[:, -1, :]

        action_logits = self.actor(out)
        return action_logits, new_hidden_state

    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size)


class REINFORCEAgent:
    def __init__(self, network):
        self.network = network
        self.hidden_state = None

    def reset_memory(self):
        self.hidden_state = self.network.init_hidden()

    def select_action(self, obs, has_key):
        """
        obs: (3, 3) numpy array
        has_key: boolean status from env
        """
        obs_tensor = torch.tensor(obs, dtype=torch.long).unsqueeze(0).unsqueeze(0)
        has_key_tensor = torch.tensor([[[float(has_key)]]], dtype=torch.float32)

        logits, self.hidden_state = self.network(obs_tensor, has_key_tensor, self.hidden_state)

        probs = F.softmax(logits, dim=-1)
        distribution = Categorical(probs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return (
            action.item(),
            log_prob,
            distribution.entropy(),
        )

    def act(self, obs, has_key, deterministic=False):
        """Run inference with explicit status injection."""
        obs_tensor = torch.tensor(obs, dtype=torch.long).unsqueeze(0).unsqueeze(0)
        has_key_tensor = torch.tensor([[[float(has_key)]]], dtype=torch.float32)

        logits, self.hidden_state = self.network(obs_tensor, has_key_tensor, self.hidden_state)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            distribution = Categorical(probs)
            action = distribution.sample()

        return action.item()