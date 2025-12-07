"""
Proximal Policy Optimization (PPO) agent for action sequence modeling.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Tuple, List, Dict


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 128]):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        shared_layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims[:-1]:
            shared_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(state)
        action_probs = self.actor(shared)
        value = self.critic(shared)
        return action_probs, value
    
    def get_action(self, state: torch.Tensor) -> Tuple[int, float, torch.Tensor]:
        """Sample action from policy."""
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.squeeze()


class PPOAgent:
    """PPO Agent for behavioral task learning."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 device: str = 'cpu'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Network
        self.network = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Training stats
        self.loss_history = []
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob, value = self.network.get_action(state_tensor)
        return action, log_prob, value.item()
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                   dones: List[bool], next_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        returns = []
        
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
            next_value = values[step]
        
        return advantages, returns
    
    def update(self, states: List[np.ndarray], actions: List[int],
               old_log_probs: List[float], rewards: List[float],
               values: List[float], dones: List[bool], next_value: float = 0.0,
               n_epochs: int = 10):
        """Update policy using PPO."""
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Training loop
        total_loss = 0
        for epoch in range(n_epochs):
            # Forward pass
            action_probs, values_pred = self.network(states_tensor)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy().mean()
            
            # Compute ratios
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            
            # Policy loss (clipped)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values_pred.squeeze(), returns_tensor)
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / n_epochs
        self.loss_history.append(avg_loss)
        
        return avg_loss
    
    def save(self, filepath: str):
        """Save model."""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
    
    def get_action_sequence(self, env, max_steps: int = 100) -> dict:
        """Generate action sequence for analysis."""
        state = env.reset()
        actions = []
        states = [state]
        rewards = []
        log_probs = []
        values = []
        
        for _ in range(max_steps):
            action, log_prob, value = self.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            actions.append(action)
            states.append(next_state)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            
            if done:
                break
            
            state = next_state
        
        return {
            'actions': np.array(actions),
            'states': np.array(states),
            'rewards': np.array(rewards),
            'log_probs': np.array(log_probs),
            'values': np.array(values),
            'total_reward': sum(rewards)
        }

