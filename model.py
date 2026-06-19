"""Policy network for Pokemon TCG RL agent."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyNet(nn.Module):
    """Actor-Critic network with action masking.
    
    Input: observation vector + action mask
    Output: action logits (masked), state value
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs, action_mask=None):
        h = self.shared(obs)
        logits = self.policy(h)
        
        if action_mask is not None:
            mask = action_mask.float()
            logits = logits * mask - 100.0 * (1.0 - mask)
        
        probs = F.softmax(logits, dim=-1)
        value = self.value(h)
        
        return probs, value
    
    def get_action_and_value(self, obs, action_mask=None, action=None):
        """Get action, log_prob, entropy, and value.
        
        If action is None, sample from the policy.
        If action is provided, compute its log_prob.
        """
        probs, value = self.forward(obs, action_mask)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)


class Agent:
    """High-level agent interface for both training and submission."""
    
    def __init__(self, policy_net: PolicyNet, device="cpu"):
        self.policy_net = policy_net
        self.device = device
    
    def act(self, obs_dict, deterministic=False):
        """Select action given observation dict from env_wrapper.
        
        Returns: list of selected option indices (length = maxCount)
        """
        num_to_select = obs_dict.get("max_count", 1)
        valid = np.where(obs_dict["action_mask"])[0]
        if len(valid) == 0:
            return []
        
        obs = torch.from_numpy(obs_dict["obs"]).float().unsqueeze(0).to(self.device)
        mask = torch.from_numpy(obs_dict["action_mask"]).bool().unsqueeze(0).to(self.device)
        
        actions = []
        remaining = valid.tolist()
        
        with torch.no_grad():
            for _ in range(num_to_select):
                if not remaining:
                    break
                current_mask = np.zeros(len(obs_dict["action_mask"]), dtype=bool)
                for r in remaining:
                    current_mask[r] = True
                mask_t = torch.from_numpy(current_mask).bool().unsqueeze(0).to(self.device)
                
                probs, _ = self.policy_net(obs, mask_t)
                mask_probs = probs.squeeze(0).cpu().numpy()
                
                if deterministic:
                    idx = remaining[mask_probs[remaining].argmax()]
                else:
                    remaining_probs = mask_probs[remaining]
                    remaining_probs = remaining_probs / remaining_probs.sum()
                    idx = remaining[np.random.choice(len(remaining), p=remaining_probs)]
                
                actions.append(idx)
                remaining.remove(idx)
        
        return actions
    
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
    
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
