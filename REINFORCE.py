import torch
import torch.nn as nn
from torch.distributions import Categorical


class REINFORCENetwork(nn.Module):
    def __init__(self, obs_shape=(3, 3), num_actions=4, hidden_size=64):
        super().__init__()

        self.hidden_size = hidden_size

        num_classes = 6
        embedding_dim = 8
        self.embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=embedding_dim)
        flattened_dim = obs_shape[0] * obs_shape[1] * embedding_dim

        self.gru = nn.GRU(input_size=flattened_dim + 1, hidden_size=hidden_size, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
        )

    def forward(self, x, has_key, hidden_state):
        batch_size, seq_len, _, _ = x.shape
        x = self.embedding(x.long()).reshape(batch_size, seq_len, -1)
        combined_features = torch.cat((x, has_key), dim=-1)
        out, new_hidden_state = self.gru(combined_features, hidden_state)
        out = out[:, -1, :]
        action_logits = self.actor(out)
        return action_logits, new_hidden_state

    def init_hidden(self, batch_size=1, device=None):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class REINFORCEAgent:
    def __init__(self, network, device=None):
        self.network = network
        self.device = device if device is not None else next(network.parameters()).device
        self.hidden_state = None

    def reset_memory(self):
        self.hidden_state = self.network.init_hidden(device=self.device)

    def _prepare_inputs(self, obs, has_key):
        obs_tensor = torch.as_tensor(obs, dtype=torch.long, device=self.device).unsqueeze(0).unsqueeze(0)
        has_key_tensor = torch.tensor([[[float(has_key)]]], dtype=torch.float32, device=self.device)
        return obs_tensor, has_key_tensor

    def select_action(self, obs, has_key):
        obs_tensor, has_key_tensor = self._prepare_inputs(obs, has_key)
        logits, self.hidden_state = self.network(obs_tensor, has_key_tensor, self.hidden_state)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action.item(), log_prob, distribution.entropy()

    def act(self, obs, has_key, deterministic=False):
        obs_tensor, has_key_tensor = self._prepare_inputs(obs, has_key)
        logits, self.hidden_state = self.network(obs_tensor, has_key_tensor, self.hidden_state)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            distribution = Categorical(logits=logits)
            action = distribution.sample()

        return action.item()
