"""
Custom Gym environment for behavioral tasks in computational neuroscience.
Simulates goal-directed decision-making tasks with neural signal alignment.
"""

import numpy as np
import gym
from gym import spaces
from typing import Dict, Tuple, Optional


class BehavioralTaskEnv(gym.Env):
    """
    A behavioral task environment simulating goal-directed decision making.
    Agents navigate a task space to reach goals, with actions tracked for
    neural-behavioral alignment analysis.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 grid_size: int = 10,
                 n_goals: int = 2,
                 max_steps: int = 100,
                 reward_scale: float = 1.0,
                 noise_level: float = 0.1):
        """
        Initialize the behavioral task environment.
        
        Args:
            grid_size: Size of the grid world
            n_goals: Number of goal locations
            max_steps: Maximum steps per episode
            reward_scale: Scaling factor for rewards
            noise_level: Noise level for observations
        """
        super(BehavioralTaskEnv, self).__init__()
        
        self.grid_size = grid_size
        self.n_goals = n_goals
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.noise_level = noise_level
        
        # Action space: 4 directions (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        
        # Observation space: agent position + goal positions + step count
        # [x, y, goal1_x, goal1_y, ..., goalN_x, goalN_y, step/max_steps]
        obs_dim = 2 + 2 * n_goals + 1
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        # Random agent position
        self.agent_pos = np.random.randint(0, self.grid_size, size=2)
        
        # Random goal positions
        self.goals = []
        for _ in range(self.n_goals):
            goal = np.random.randint(0, self.grid_size, size=2)
            # Ensure goal is not at agent position
            while np.array_equal(goal, self.agent_pos):
                goal = np.random.randint(0, self.grid_size, size=2)
            self.goals.append(goal)
        
        self.step_count = 0
        self.action_history = []
        self.reward_history = []
        self.visited_positions = [self.agent_pos.copy()]
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation with optional noise."""
        obs = []
        
        # Normalized agent position
        obs.extend((self.agent_pos / self.grid_size) * 2 - 1)
        
        # Normalized goal positions
        for goal in self.goals:
            obs.extend((goal / self.grid_size) * 2 - 1)
        
        # Normalized step count
        obs.append(self.step_count / self.max_steps)
        
        obs = np.array(obs, dtype=np.float32)
        
        # Add noise if specified
        if self.noise_level > 0:
            obs += np.random.normal(0, self.noise_level, obs.shape)
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment."""
        self.step_count += 1
        
        # Action mapping: 0=up, 1=down, 2=left, 3=right
        action_map = {
            0: np.array([0, -1]),  # up
            1: np.array([0, 1]),   # down
            2: np.array([-1, 0]),  # left
            3: np.array([1, 0])    # right
        }
        
        # Update agent position
        movement = action_map[action]
        new_pos = self.agent_pos + movement
        
        # Clip to grid boundaries
        new_pos = np.clip(new_pos, 0, self.grid_size - 1)
        self.agent_pos = new_pos
        self.visited_positions.append(self.agent_pos.copy())
        self.action_history.append(action)
        
        # Calculate reward
        reward = self._calculate_reward()
        self.reward_history.append(reward)
        
        # Check termination
        done = self._check_termination()
        
        # Info dictionary for analysis
        info = {
            'agent_pos': self.agent_pos.copy(),
            'goals': [g.copy() for g in self.goals],
            'action': action,
            'step': self.step_count,
            'reached_goal': self._check_goal_reached()
        }
        
        return self._get_observation(), reward, done, info
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on distance to goals."""
        # Distance-based reward (closer = better)
        min_dist = float('inf')
        for goal in self.goals:
            dist = np.linalg.norm(self.agent_pos - goal)
            min_dist = min(min_dist, dist)
        
        # Reward for being close to goal
        reward = -min_dist / self.grid_size
        
        # Bonus for reaching goal
        if self._check_goal_reached():
            reward += 10.0 * self.reward_scale
        
        # Small penalty for each step
        reward -= 0.01 * self.reward_scale
        
        return reward * self.reward_scale
    
    def _check_goal_reached(self) -> bool:
        """Check if agent reached any goal."""
        for goal in self.goals:
            if np.array_equal(self.agent_pos, goal):
                return True
        return False
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        if self._check_goal_reached():
            return True
        if self.step_count >= self.max_steps:
            return True
        return False
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            grid = np.zeros((self.grid_size, self.grid_size))
            
            # Mark goals
            for i, goal in enumerate(self.goals):
                grid[goal[1], goal[0]] = 2 + i
            
            # Mark agent
            grid[self.agent_pos[1], self.agent_pos[0]] = 1
            
            print("\n" + "=" * (self.grid_size * 2 + 1))
            for row in grid:
                print("|", end="")
                for cell in row:
                    if cell == 0:
                        print(" ", end="|")
                    elif cell == 1:
                        print("A", end="|")
                    else:
                        print(f"G{int(cell-2)}", end="|")
                print()
            print("=" * (self.grid_size * 2 + 1))
            print(f"Step: {self.step_count}/{self.max_steps}")
        
        return None
    
    def get_trajectory(self) -> Dict:
        """Get full trajectory for analysis."""
        return {
            'positions': np.array(self.visited_positions),
            'actions': np.array(self.action_history),
            'rewards': np.array(self.reward_history),
            'goals': np.array(self.goals),
            'steps': self.step_count
        }


class MultiAgentBehavioralEnv(gym.Env):
    """
    Multi-agent version for analyzing action sequences across agents.
    """
    
    def __init__(self, 
                 n_agents: int = 3,
                 grid_size: int = 10,
                 n_goals: int = 2,
                 max_steps: int = 100):
        super(MultiAgentBehavioralEnv, self).__init__()
        
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.n_goals = n_goals
        self.max_steps = max_steps
        
        # Action space: joint actions of all agents
        self.action_space = spaces.MultiDiscrete([4] * n_agents)
        
        # Observation space: positions of all agents + goals
        obs_dim = 2 * n_agents + 2 * n_goals + 1
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment with multiple agents."""
        self.agent_positions = []
        for _ in range(self.n_agents):
            pos = np.random.randint(0, self.grid_size, size=2)
            self.agent_positions.append(pos)
        
        self.goals = []
        for _ in range(self.n_goals):
            goal = np.random.randint(0, self.grid_size, size=2)
            self.goals.append(goal)
        
        self.step_count = 0
        self.trajectories = [[] for _ in range(self.n_agents)]
        self.action_sequences = [[] for _ in range(self.n_agents)]
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get joint observation."""
        obs = []
        for pos in self.agent_positions:
            obs.extend((pos / self.grid_size) * 2 - 1)
        for goal in self.goals:
            obs.extend((goal / self.grid_size) * 2 - 1)
        obs.append(self.step_count / self.max_steps)
        return np.array(obs, dtype=np.float32)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute joint actions."""
        self.step_count += 1
        
        action_map = {
            0: np.array([0, -1]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([1, 0])
        }
        
        rewards = []
        for i, action in enumerate(actions):
            movement = action_map[action]
            new_pos = self.agent_positions[i] + movement
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)
            self.agent_positions[i] = new_pos
            self.trajectories[i].append(new_pos.copy())
            self.action_sequences[i].append(action)
            
            # Individual reward
            min_dist = min([np.linalg.norm(new_pos - g) for g in self.goals])
            reward = -min_dist / self.grid_size
            if any([np.array_equal(new_pos, g) for g in self.goals]):
                reward += 10.0
            rewards.append(reward)
        
        total_reward = np.mean(rewards)
        done = self.step_count >= self.max_steps or all([
            any([np.array_equal(pos, g) for g in self.goals])
            for pos in self.agent_positions
        ])
        
        info = {
            'agent_positions': [p.copy() for p in self.agent_positions],
            'action_sequences': [a.copy() for a in self.action_sequences],
            'individual_rewards': rewards
        }
        
        return self._get_observation(), total_reward, done, info
    
    def get_multi_agent_trajectories(self) -> Dict:
        """Get trajectories for all agents."""
        return {
            'trajectories': [np.array(t) for t in self.trajectories],
            'action_sequences': [np.array(a) for a in self.action_sequences],
            'goals': np.array(self.goals)
        }

