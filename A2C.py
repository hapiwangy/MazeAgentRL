import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class A2CNetwork(nn.Module):
    def __init__(self, obs_shape=(3, 3), num_actions=4, hidden_size=64):
        super(A2CNetwork, self).__init__()

        self.hidden_size = hidden_size

        # 1. Observation encoder
        # Map grid values (0-5) to continuous vectors
        num_classes = 6  
        embedding_dim = 8
        self.embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=embedding_dim)

        # Flattened dimension: 3 * 3 * 8 = 72
        flattened_dim = obs_shape[0] * obs_shape[1] * embedding_dim

        # 2. Recurrent layer (GRU)
        # We add +1 to the input size to account for the 'has_key' boolean flag
        self.gru = nn.GRU(input_size=flattened_dim + 1, hidden_size=hidden_size, batch_first=True)

        # 3. Actor network (Policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
        )

        # 4. Critic network (Value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, has_key, hidden_state):
        """
        x: (batch_size, seq_len, 3, 3)
        has_key: (batch_size, seq_len, 1) - float tensor (0.0 or 1.0)
        hidden_state: (1, batch_size, hidden_size)
        """
        batch_size, seq_len, h, w = x.shape

        # Embed and flatten the grid
        x = self.embedding(x.long()) # (batch, seq, 3, 3, 8)
        x = x.view(batch_size, seq_len, -1) # (batch, seq, 72)

        # Concatenate the 'has_key' status to the features
        # This gives the GRU explicit context: "I am in key-seeking mode" vs "I am in exit-seeking mode"
        combined_features = torch.cat((x, has_key), dim=-1) # (batch, seq, 73)

        # Pass through GRU
        out, new_hidden_state = self.gru(combined_features, hidden_state)

        # Use the output from the last time step
        out = out[:, -1, :]

        action_logits = self.actor(out)
        state_value = self.critic(out)

        return action_logits, state_value, new_hidden_state

    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size)


class A2CAgent:
    def __init__(self, network):
        self.network = network
        self.hidden_state = None

    def reset_memory(self):
        """Reset recurrent memory at the start of each episode."""
        self.hidden_state = self.network.init_hidden()

    def select_action(self, obs, has_key):
        """
        obs: (3, 3) numpy array
        has_key: bool
        """
        # Convert inputs to tensors with (batch=1, seq=1) dimensions
        obs_tensor = torch.tensor(obs, dtype=torch.long).unsqueeze(0).unsqueeze(0)
        has_key_tensor = torch.tensor([[[float(has_key)]]], dtype=torch.float32)

        # Forward pass
        logits, state_value, self.hidden_state = self.network(
            obs_tensor, has_key_tensor, self.hidden_state
        )

        # Action selection
        probs = F.softmax(logits, dim=-1)
        distribution = Categorical(probs)
        action = distribution.sample()

        log_prob = distribution.log_prob(action)

        return (
            action.item(),
            log_prob.unsqueeze(0),
            state_value.view(1),
            distribution.entropy().unsqueeze(0),
        )

    def act(self, obs, has_key, deterministic=False):
        """Inference mode."""
        obs_tensor = torch.tensor(obs, dtype=torch.long).unsqueeze(0).unsqueeze(0)
        has_key_tensor = torch.tensor([[[float(has_key)]]], dtype=torch.float32)

        logits, _, self.hidden_state = self.network(
            obs_tensor, has_key_tensor, self.hidden_state
        )

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            distribution = Categorical(probs)
            action = distribution.sample()

        return action.item()