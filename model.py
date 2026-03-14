"""
model.py — The Biological Brain for BioDrone-RL

Implements a sparse neural network inspired by the Drosophila connectome.
MaskedLinear layers permanently sever a fraction of connections (synapses),
mimicking the fixed wiring of a fruit fly's nervous system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ------------------------------------------------------------------
# Device Setup — Apple Silicon MPS or CPU fallback
# ------------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Apple Silicon MPS backend detected — running on M1 GPU")
    else:
        device = torch.device("cpu")
        print("⚠️  MPS not available — falling back to CPU")
    return device


# ------------------------------------------------------------------
# MaskedLinear — The Connectome Layer
# ------------------------------------------------------------------

class MaskedLinear(nn.Module):
    """
    A biologically-inspired linear layer where a fixed boolean mask
    permanently severs a fraction of synaptic connections.

    Think of it as a fruit fly's connectome — the wiring is fixed at
    'birth' (initialization) and never changes during the fly's lifetime.
    Only the *strength* of surviving connections (weights) can be learned.

    Args:
        in_features  (int)   : Number of input neurons
        out_features (int)   : Number of output neurons
        sparsity     (float) : Fraction of connections to sever (0.0 - 1.0)
                               e.g. 0.8 means 80% of synapses are dead
        bias         (bool)  : Whether to include a bias term
    """

    def __init__(self, in_features, out_features, sparsity=0.8, bias=True):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features
        self.sparsity     = sparsity

        # --- Learnable weight parameter (standard linear layer weights) ---
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # --- Initialize weights using Kaiming uniform (good for ReLU) ---
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

        # --- Build the permanent connectome mask ---
        # register_buffer: saves with model state_dict but is NOT a parameter
        # (won't be updated by the optimizer — it's fixed forever)
        mask = self._build_mask(out_features, in_features, sparsity)
        self.register_buffer('mask', mask)

        # --- Zero out the severed weights at initialization ---
        # Not strictly necessary (mask will zero them in forward),
        # but keeps the weight tensor clean and inspectable
        with torch.no_grad():
            self.weight.data *= self.mask

    def _build_mask(self, out_features, in_features, sparsity):
        """
        Create a binary mask where:
          - 0 = severed synapse (permanently dead connection)
          - 1 = active synapse  (connection that can carry signal)

        The mask is sampled randomly once at init and never changes.
        """
        total_connections = out_features * in_features
        num_active = int(total_connections * (1.0 - sparsity))

        # Start with all zeros (all severed)
        mask = torch.zeros(out_features * in_features)

        # Randomly activate (1.0 - sparsity) fraction of connections
        active_indices = torch.randperm(total_connections)[:num_active]
        mask[active_indices] = 1.0

        return mask.reshape(out_features, in_features)

    def forward(self, x):
        """
        Forward pass through the sparse connectome.

        The key operation: weight * mask
        - Severed connections (mask=0) produce zero output regardless
          of their weight value
        - Because mask is a constant buffer (not a parameter), PyTorch's
          autograd will naturally zero out gradients for severed connections
          during backpropagation — they can never learn
        """
        sparse_weight = self.weight * self.mask
        return F.linear(x, sparse_weight, self.bias)

    def extra_repr(self):
        """Clean string representation for print(model)"""
        active = int(self.mask.sum().item())
        total  = self.in_features * self.out_features
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"sparsity={self.sparsity:.0%}, "
            f"active_synapses={active}/{total}"
        )


# ------------------------------------------------------------------
# FlyPolicyNetwork — The Fruit Fly Brain
# ------------------------------------------------------------------

class FlyPolicyNetwork(nn.Module):
    """
    A sparse policy network inspired by the Drosophila melanogaster connectome.

    Architecture:
        Input  (5)  — 5 distance sensors
           ↓
        MaskedLinear(5 → 32, sparsity=0.8)  — sparse hidden layer
           ↓
        ReLU                                 — biological activation
           ↓
        MaskedLinear(32 → 3, sparsity=0.8)  — sparse output layer
           ↓
        Softmax                              — action probability distribution

    The network outputs a Categorical distribution over 3 actions:
        [P(Steer Left), P(Go Straight), P(Steer Right)]

    Args:
        input_dim  (int)   : Number of sensor inputs (default: 5)
        hidden_dim (int)   : Hidden layer size (default: 32)
        output_dim (int)   : Number of actions (default: 3)
        sparsity   (float) : Fraction of dead connections (default: 0.8)
    """

    def __init__(self, input_dim=5, hidden_dim=32, output_dim=3, sparsity=0.8):
        super().__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sparsity   = sparsity

        # --- Sparse Connectome Layers ---
        self.fc1 = MaskedLinear(input_dim,  hidden_dim, sparsity=sparsity)
        self.fc2 = MaskedLinear(hidden_dim, output_dim, sparsity=sparsity)

    def forward(self, x):
        """
        Forward pass: sensor readings → action probability distribution.

        Args:
            x (torch.Tensor): Shape (batch_size, 5) or (5,) — sensor observations

        Returns:
            dist (Categorical): A probability distribution over actions.
                                Sample with dist.sample() or get log probs
                                with dist.log_prob(action)
        """
        # Ensure input is at least 2D for batch compatibility
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Layer 1: Sparse synaptic transmission + ReLU (biological firing)
        x = F.relu(self.fc1(x))

        # Layer 2: Sparse output projection
        logits = self.fc2(x)

        # Softmax → probability distribution over actions
        probs = F.softmax(logits, dim=-1)

        # Return a Categorical distribution object
        # This lets the training loop call:
        #   action = dist.sample()
        #   log_prob = dist.log_prob(action)
        dist = Categorical(probs)
        return dist

    def get_action(self, obs_tensor):
        """
        Convenience method: given a raw observation tensor,
        sample an action and return it with its log probability.

        Args:
            obs_tensor (torch.Tensor): Shape (5,) — single observation

        Returns:
            action   (int)            : The chosen action (0, 1, or 2)
            log_prob (torch.Tensor)   : Log probability of that action
        """
        dist   = self.forward(obs_tensor)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


# ------------------------------------------------------------------
# Test / Introspection Script
# ------------------------------------------------------------------

def test_model():
    print("=" * 60)
    print("  BioDrone-RL | Biological Brain Inspection")
    print("=" * 60)

    # --- Device ---
    device = get_device()

    # --- Build the network ---
    model = FlyPolicyNetwork(
        input_dim=5,
        hidden_dim=32,
        output_dim=3,
        sparsity=0.8
    ).to(device)

    # --- Print architecture ---
    print("\n📐 Network Architecture:")
    print(model)
    print()

    # --- Connectome statistics ---
    total_params  = sum(p.numel() for p in model.parameters())
    active_syn_1  = int(model.fc1.mask.sum().item())
    active_syn_2  = int(model.fc2.mask.sum().item())
    total_syn_1   = model.fc1.in_features * model.fc1.out_features
    total_syn_2   = model.fc2.in_features * model.fc2.out_features

    print("🧠 Connectome Statistics:")
    print(f"   Layer 1 active synapses : {active_syn_1:>4} / {total_syn_1}  "
          f"({active_syn_1/total_syn_1:.0%} alive)")
    print(f"   Layer 2 active synapses : {active_syn_2:>4} / {total_syn_2}  "
          f"({active_syn_2/total_syn_2:.0%} alive)")
    print(f"   Total learnable params  : {total_params}")
    print()

    # --- Forward pass test ---
    print("🔬 Forward Pass Test:")
    dummy_obs = torch.rand(5).to(device)   # Simulate one sensor reading
    print(f"   Input observation : {dummy_obs.cpu().numpy().round(3)}")

    dist = model(dummy_obs)
    print(f"   Action probs      : {dist.probs.detach().cpu().numpy().round(3)}")

    action, log_prob = model.get_action(dummy_obs)
    action_names = ["Steer Left", "Go Straight", "Steer Right"]
    print(f"   Sampled action    : {action} ({action_names[action]})")
    print(f"   Log probability   : {log_prob.item():.4f}")
    print()

    # --- Gradient flow test ---
    print("⚡ Gradient Flow Test (verifying severed synapses don't learn):")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = -log_prob   # Simple dummy loss
    loss.backward()

    # Check that gradients are zero where mask is zero
    grad_fc1    = model.fc1.weight.grad
    mask_fc1    = model.fc1.mask
    severed_grads = grad_fc1[mask_fc1 == 0].abs().sum().item()
    print(f"   Gradient sum through severed synapses (should be 0.0): "
          f"{severed_grads:.6f}")

    if severed_grads < 1e-6:
        print("   ✅ Confirmed: severed synapses have zero gradients")
    else:
        print("   ❌ Warning: gradients leaking through severed synapses!")

    optimizer.zero_grad()

    print()
    print("=" * 60)
    print("  Brain check complete. Ready for Phase 3!")
    print("=" * 60)


if __name__ == "__main__":
    test_model()