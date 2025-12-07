"""
Visualization and summary tools for interpreting patterns across trials, subjects, and conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path


class BehavioralVisualizer:
    """Visualization tools for behavioral data analysis."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        plt.style.use(style)
        self.figsize = figsize
        sns.set_palette("husl")
    
    def plot_action_sequences(self,
                             action_sequences: List[np.ndarray],
                             labels: Optional[List[str]] = None,
                             title: str = "Action Sequences",
                             save_path: Optional[str] = None):
        """
        Plot action sequences across agents or trials.
        
        Args:
            action_sequences: List of action arrays
            labels: Optional labels for each sequence
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, actions in enumerate(action_sequences):
            label = labels[i] if labels else f'Sequence {i+1}'
            ax.plot(actions, marker='o', label=label, alpha=0.7)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Action')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_trajectories(self,
                         trajectories: List[np.ndarray],
                         goals: Optional[np.ndarray] = None,
                         labels: Optional[List[str]] = None,
                         title: str = "Agent Trajectories",
                         save_path: Optional[str] = None):
        """
        Plot agent trajectories in 2D space.
        
        Args:
            trajectories: List of trajectory arrays (N x 2)
            goals: Optional goal positions
            labels: Optional labels
            title: Plot title
            save_path: Optional save path
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, traj in enumerate(trajectories):
            label = labels[i] if labels else f'Agent {i+1}'
            ax.plot(traj[:, 0], traj[:, 1], marker='o', label=label, alpha=0.7, markersize=4)
            ax.scatter(traj[0, 0], traj[0, 1], marker='s', s=100, color=ax.lines[-1].get_color(), 
                      edgecolors='black', zorder=5)
            ax.scatter(traj[-1, 0], traj[-1, 1], marker='*', s=200, color=ax.lines[-1].get_color(),
                      edgecolors='black', zorder=5)
        
        if goals is not None:
            ax.scatter(goals[:, 0], goals[:, 1], marker='X', s=300, c='red', 
                      label='Goals', zorder=5, edgecolors='black')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_reward_curves(self,
                          reward_histories: List[np.ndarray],
                          labels: Optional[List[str]] = None,
                          title: str = "Learning Curves",
                          save_path: Optional[str] = None):
        """
        Plot reward curves across training.
        
        Args:
            reward_histories: List of reward arrays
            labels: Optional labels
            title: Plot title
            save_path: Optional save path
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, rewards in enumerate(reward_histories):
            label = labels[i] if labels else f'Agent {i+1}'
            # Smooth with moving average
            window = max(1, len(rewards) // 50)
            if window > 1:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                x = np.arange(window-1, len(rewards))
            else:
                smoothed = rewards
                x = np.arange(len(rewards))
            
            ax.plot(x, smoothed, label=label, alpha=0.7)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_condition_comparison(self,
                                 data: pd.DataFrame,
                                 value_col: str,
                                 condition_col: str = 'condition',
                                 title: str = "Condition Comparison",
                                 save_path: Optional[str] = None):
        """
        Compare values across conditions.
        
        Args:
            data: DataFrame with data
            value_col: Column name for values
            condition_col: Column name for conditions
            title: Plot title
            save_path: Optional save path
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Box plot
        sns.boxplot(data=data, x=condition_col, y=value_col, ax=axes[0])
        axes[0].set_title(f'{title} - Distribution')
        axes[0].set_ylabel(value_col)
        
        # Violin plot
        sns.violinplot(data=data, x=condition_col, y=value_col, ax=axes[1])
        axes[1].set_title(f'{title} - Density')
        axes[1].set_ylabel(value_col)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_subject_summary(self,
                            data: pd.DataFrame,
                            subject_col: str = 'subject_id',
                            value_col: str = 'total_reward',
                            title: str = "Subject Performance",
                            save_path: Optional[str] = None):
        """
        Plot summary across subjects.
        
        Args:
            data: DataFrame with data
            subject_col: Column name for subjects
            value_col: Column name for values
            title: Plot title
            save_path: Optional save path
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        subject_means = data.groupby(subject_col)[value_col].mean()
        subject_stds = data.groupby(subject_col)[value_col].std()
        
        x = np.arange(len(subject_means))
        ax.bar(x, subject_means.values, yerr=subject_stds.values, 
              capsize=5, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Subject ID')
        ax.set_ylabel(value_col)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(subject_means.index)
        ax.grid(True, alpha=0.3, axis='y')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()


class NeuralVisualizer:
    """Visualization tools for neural signal analysis."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6)):
        plt.style.use(style)
        self.figsize = figsize
    
    def plot_event_triggered_average(self,
                                    eta_data: Dict,
                                    title: str = "Event-Triggered Average",
                                    save_path: Optional[str] = None):
        """
        Plot event-triggered average.
        
        Args:
            eta_data: Dictionary from EventTriggeredFeatureExtractor
            title: Plot title
            save_path: Optional save path
        """
        if eta_data['eta'] is None:
            print("No valid events found for ETA")
            return
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        eta = eta_data['eta']
        time_axis = eta_data['time_axis']
        std = eta_data['std']
        
        if eta.ndim == 1:
            ax.plot(time_axis, eta, linewidth=2, label='ETA')
            if std is not None:
                ax.fill_between(time_axis, eta - std, eta + std, alpha=0.3)
        else:
            for i in range(eta.shape[1]):
                ax.plot(time_axis, eta[:, i], linewidth=2, label=f'Channel {i+1}')
        
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Event')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Signal Amplitude')
        ax.set_title(f'{title} (n={eta_data["n_events"]} events)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_neural_behavior_alignment(self,
                                      neural_signal: np.ndarray,
                                      action_times: np.ndarray,
                                      action_values: np.ndarray,
                                      time_axis: np.ndarray,
                                      title: str = "Neural-Behavioral Alignment",
                                      save_path: Optional[str] = None):
        """
        Plot neural signal aligned with behavioral actions.
        
        Args:
            neural_signal: Neural signal
            action_times: Action times
            action_values: Action values
            time_axis: Time axis
            title: Plot title
            save_path: Optional save path
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Neural signal
        axes[0].plot(time_axis, neural_signal, linewidth=1, alpha=0.7, label='Neural Signal')
        for action_time, action_val in zip(action_times, action_values):
            axes[0].axvline(x=action_time, color='red', linestyle='--', alpha=0.5)
            axes[0].text(action_time, axes[0].get_ylim()[1] * 0.9, 
                        f'A{action_val}', rotation=90, fontsize=8)
        axes[0].set_ylabel('Neural Signal')
        axes[0].set_title(title)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Action timeline
        action_colors = plt.cm.tab10(action_values / max(action_values))
        for i, (action_time, action_val) in enumerate(zip(action_times, action_values)):
            axes[1].barh(0, 0.1, left=action_time, height=0.5, 
                        color=action_colors[i], alpha=0.7, edgecolor='black')
            axes[1].text(action_time, 0, f'{action_val}', ha='center', va='center', fontsize=8)
        
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Actions')
        axes[1].set_ylim(-0.5, 0.5)
        axes[1].set_yticks([])
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_trial_summary(self,
                          trial_data: Dict,
                          title: str = "Trial Summary",
                          save_path: Optional[str] = None):
        """
        Plot comprehensive trial summary.
        
        Args:
            trial_data: Dictionary with trial data
            title: Plot title
            save_path: Optional save path
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        time_axis = trial_data['time_axis']
        neural_signal = trial_data['neural_signal']
        action_times = trial_data['action_times']
        actions = trial_data['actions']
        rewards = trial_data['rewards']
        
        # Neural signal
        axes[0].plot(time_axis, neural_signal, linewidth=1.5, color='blue', alpha=0.7)
        axes[0].set_ylabel('Neural Signal')
        axes[0].set_title(title)
        axes[0].grid(True, alpha=0.3)
        
        # Actions
        for action_time, action in zip(action_times, actions):
            axes[1].scatter(action_time, action, s=100, alpha=0.7, edgecolors='black')
        axes[1].set_ylabel('Action')
        axes[1].set_ylim(-0.5, 3.5)
        axes[1].set_yticks([0, 1, 2, 3])
        axes[1].grid(True, alpha=0.3)
        
        # Rewards
        for action_time, reward in zip(action_times, rewards):
            color = 'green' if reward > 0 else 'red'
            axes[2].bar(action_time, reward, width=0.1, color=color, alpha=0.7, edgecolor='black')
        axes[2].set_ylabel('Reward')
        axes[2].set_xlabel('Time (s)')
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()


class SummaryGenerator:
    """Generate summary statistics and reports."""
    
    @staticmethod
    def generate_trial_summary(trial_data: Dict) -> str:
        """Generate text summary for a trial."""
        summary = f"""
Trial Summary
=============
Trial ID: {trial_data.get('trial_id', 'N/A')}
Subject ID: {trial_data.get('subject_id', 'N/A')}
Condition: {trial_data.get('condition', 'N/A')}
Trial Type: {trial_data.get('trial_type', 'N/A')}

Behavioral Metrics:
- Number of Actions: {trial_data.get('n_actions', len(trial_data.get('actions', [])))}
- Total Reward: {trial_data.get('total_reward', 0):.2f}
- Mean Reaction Time: {np.mean(trial_data.get('reaction_times', [0])):.3f} s

Neural Metrics:
- Signal Duration: {len(trial_data.get('neural_signal', []))} samples
- Signal Mean: {np.mean(trial_data.get('neural_signal', [0])):.3f}
- Signal Std: {np.std(trial_data.get('neural_signal', [0])):.3f}
"""
        return summary
    
    @staticmethod
    def generate_dataset_summary(df: pd.DataFrame, output_path: Optional[str] = None) -> str:
        """Generate summary statistics for entire dataset."""
        summary_lines = [
            "Dataset Summary",
            "=" * 50,
            f"Total Trials: {len(df)}",
            f"Number of Subjects: {df['subject_id'].nunique() if 'subject_id' in df.columns else 'N/A'}",
            f"Number of Conditions: {df['condition'].nunique() if 'condition' in df.columns else 'N/A'}",
            ""
        ]
        
        if 'total_reward' in df.columns:
            summary_lines.extend([
                "Reward Statistics:",
                f"  Mean: {df['total_reward'].mean():.2f}",
                f"  Std: {df['total_reward'].std():.2f}",
                f"  Min: {df['total_reward'].min():.2f}",
                f"  Max: {df['total_reward'].max():.2f}",
                ""
            ])
        
        if 'condition' in df.columns and 'total_reward' in df.columns:
            summary_lines.append("Reward by Condition:")
            for condition in sorted(df['condition'].unique()):
                cond_data = df[df['condition'] == condition]['total_reward']
                summary_lines.append(
                    f"  Condition {condition}: {cond_data.mean():.2f} ± {cond_data.std():.2f}"
                )
            summary_lines.append("")
        
        summary = "\n".join(summary_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(summary)
        
        return summary

