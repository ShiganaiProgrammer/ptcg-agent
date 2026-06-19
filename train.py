"""PPO training for Pokemon TCG RL agent."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from collections import deque

from env_wrapper import CabtEnv, MAX_OPTIONS
from model import PolicyNet, Agent


class RolloutBuffer:
    """Stores trajectory data for PPO updates."""
    
    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.action_masks = []
    
    def add(self, obs, action, log_prob, reward, done, value, action_mask):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.action_masks.append(action_mask)
    
    def get(self):
        return (
            torch.from_numpy(np.array(self.observations)).float(),
            torch.from_numpy(np.array(self.actions)).long(),
            torch.from_numpy(np.array(self.log_probs)).float(),
            torch.from_numpy(np.array(self.rewards)).float(),
            torch.from_numpy(np.array(self.dones)).float(),
            torch.from_numpy(np.array(self.values)).float(),
            torch.from_numpy(np.array(self.action_masks)).bool(),
        )
    
    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.action_masks.clear()
    
    def __len__(self):
        return len(self.observations)


class PPO:
    """PPO trainer with clipped objective and GAE."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = MAX_OPTIONS,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        device: str = "cpu",
    ):
        self.policy_net = PolicyNet(obs_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.device = device
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        
        self.agent = Agent(self.policy_net, device)
        self.buffer = RolloutBuffer()
        self.steps = 0
    
    def collect_rollout(self, env: CabtEnv, n_steps: int):
        """Collect n_steps of experience from environment."""
        self.buffer.clear()
        obs_dict = env.reset()
        episode_rewards = []
        episode_length = 0
        
        for _ in range(n_steps):
            num_to_select = obs_dict.get("max_count", 1)
            valid = np.where(obs_dict["action_mask"])[0]
            
            if len(valid) == 0:
                next_obs_dict, reward, done, _ = env.step([])
                self.buffer.add(obs_dict["obs"].copy(), -1, 0.0, reward, done, 0.0, obs_dict["action_mask"].copy())
                if done:
                    obs_dict = env.reset()
                else:
                    obs_dict = next_obs_dict
                continue
            
            actions = []
            total_log_prob = 0.0
            remaining = valid.tolist()
            
            obs_t = torch.from_numpy(obs_dict["obs"]).float().unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                value = self.policy_net.shared(obs_t)
                value = self.policy_net.value(value).squeeze(-1).item()
                
                for _ in range(num_to_select):
                    if not remaining:
                        break
                    current_mask = np.zeros(len(obs_dict["action_mask"]), dtype=bool)
                    for r in remaining:
                        current_mask[r] = True
                    
                    mask_t = torch.from_numpy(current_mask).bool().unsqueeze(0).to(self.device)
                    probs, _ = self.policy_net(obs_t, mask_t)
                    dist = torch.distributions.Categorical(probs)
                    
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    total_log_prob += log_prob
                    
                    idx = action.item()
                    if idx >= len(remaining):
                        idx = remaining[-1]
                    else:
                        idx = remaining[idx % len(remaining)]
                    
                    actions.append(idx)
                    remaining.remove(idx)
            
            next_obs_dict, reward, done, _ = env.step(actions)
            
            self.buffer.add(
                obs_dict["obs"].copy(),
                actions[0] if actions else -1,
                total_log_prob.item(),
                reward,
                done,
                value,
                obs_dict["action_mask"].copy(),
            )
            
            episode_rewards.append(reward)
            episode_length += 1
            
            if done:
                obs_dict = env.reset()
            else:
                obs_dict = next_obs_dict
        
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        last_values = [r for r in episode_rewards if r != 0]
        win_rate = sum(1 for r in last_values if r > 0) / max(len(last_values), 1)
        
        return {
            "avg_reward": float(avg_reward),
            "episode_length": episode_length,
            "win_rate": float(win_rate),
        }
    
    def compute_gae(self, rewards, values, dones, last_value=0.0):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0.0
        values = values.tolist() + [last_value]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float32)
        return advantages, returns
    
    def update(self):
        """Update policy using collected rollout."""
        obs, actions, old_log_probs, rewards, dones, values, action_masks = self.buffer.get()
        
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        action_masks = action_masks.to(self.device)
        
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        for _ in range(self.update_epochs):
            probs, values_pred = self.policy_net(obs, action_masks)
            dist = torch.distributions.Categorical(probs)
            
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values_pred.squeeze(-1), returns)
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
        
        n_updates = self.update_epochs
        return {
            "loss": total_loss / n_updates,
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "advantages_mean": advantages.mean().item(),
        }
    
    def train(
        self,
        env: CabtEnv,
        total_steps: int = 100_000,
        rollout_steps: int = 512,
        log_interval: int = 10,
        eval_interval: int = 50,
        eval_episodes: int = 10,
        save_path: str = "checkpoints/model.pt",
    ):
        """Main training loop."""
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        
        pbar = tqdm(total=total_steps, desc="Training")
        stats_history = deque(maxlen=100)
        
        while self.steps < total_steps:
            rollout_info = self.collect_rollout(env, rollout_steps)
            self.steps += rollout_steps
            pbar.update(rollout_steps)
            
            update_info = self.update()
            
            stats_history.append(rollout_info["avg_reward"])
            avg_rew = np.mean(stats_history)
            
            if self.steps % (rollout_steps * log_interval) < rollout_steps:
                pbar.set_postfix({
                    "rew": f"{rollout_info['avg_reward']:.3f}",
                    "win": f"{rollout_info['win_rate']:.2f}",
                    "loss": f"{update_info['loss']:.3f}",
                    "ent": f"{update_info['entropy']:.3f}",
                })
            
            if self.steps % (rollout_steps * eval_interval) < rollout_steps:
                eval_info = self.evaluate(env, eval_episodes)
                print(f"\n[Eval @ {self.steps}] Win rate: {eval_info['win_rate']:.3f}, "
                      f"Avg reward: {eval_info['avg_reward']:.3f}")
                self.agent.save(f"{save_path}.{self.steps}.pt")
        
        pbar.close()
        self.agent.save(save_path)
        return self.agent
    
    def evaluate(self, env: CabtEnv, n_episodes: int = 10):
        """Evaluate current policy."""
        rewards = []
        for _ in range(n_episodes):
            obs_dict = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action = self.agent.act(obs_dict, deterministic=True)
                obs_dict, reward, done, _ = env.step(action)
                total_reward += reward
            rewards.append(total_reward)
        
        wins = sum(1 for r in rewards if r > 0)
        return {
            "avg_reward": float(np.mean(rewards)),
            "win_rate": wins / max(n_episodes, 1),
        }
