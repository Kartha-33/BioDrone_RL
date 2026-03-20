import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Linear):
    """
    Linear layer with a binary mask to enforce coordinate-sparse connectivity.
    """
    def __init__(self, in_features, out_features, mask=None, density=0.5):
        super().__init__(in_features, out_features)
        
        if mask is None:
            # Fallback to random if no explicit mask provided
            mask = torch.rand(out_features, in_features) < density
            
        self.register_buffer('mask', mask.float())

    def forward(self, input):
        masked_weight = self.weight * self.mask
        return F.linear(input, masked_weight, self.bias)

class SparsePolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        # Input standard Linear, Hidden is Masked (Bio-mimetic placeholder)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bio_layer = MaskedLinear(hidden_dim, hidden_dim, density=0.3)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.bio_layer(x))
        logits = self.fc_out(x)
        return F.softmax(logits, dim=-1)

class VisionPolicy(nn.Module):
    """
    Phase B: Vision-Based Policy.
    Input: 1D Visual Vector (Retina)
    Hidden: Bio-Masked Layer
    Output: Action logits
    """
    def __init__(self, input_dim=32, output_dim=3, hidden_dim=64):
        super().__init__()
        # Input layer (Processing the 'Retina')
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # The 'Biological' processing layer (Masked)
        self.bio_layer = MaskedLinear(hidden_dim, hidden_dim, density=0.4)
        
        # Decision layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.bio_layer(x))
        logits = self.fc_out(x)
        return F.softmax(logits, dim=-1)

class ActorCritic(nn.Module):
    """
    Phase C: Continuous PPO Actor-Critic.
    Supports Bio-Constrained layers in the shared trunk or actor head.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64, mask_type='random'):
        super().__init__()
        
        # Example: Feature Extractor with Connectome Constraints
        # Layer 1: Retinotopic (Local) connections
        # Mimics the eye-to-brain pathway
        self.features = nn.Sequential(
            MaskedLinear(state_dim, hidden_dim, mask=None, density=0.3), # We will inject mask later
            nn.Tanh(),
            # Layer 2: Fully connected integration (Medulla -> Lobula)
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor Head (Mean of action distribution)
        # Using Tanh to bound actions to [-1, 1]
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Actor Log Std (Learnable parameter)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic Head (Value function)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        # Extract features
        x = self.features(state)
        
        # Actor & Critic
        action_mean = self.actor_mean(x)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        value = self.critic(x)
        
        return action_mean, action_std, value

    def get_action(self, state):
        # Convert state to tensor if needed
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
            
        action_mean, action_std, value = self.forward(state)
        
        # Create Normal Distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        
        return action, log_prob, value

    def evaluate(self, state, action):
        action_mean, action_std, value = self.forward(state)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        
        return log_prob, value, entropy