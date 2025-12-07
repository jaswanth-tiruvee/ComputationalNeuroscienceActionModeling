"""
Simulation-based dataset generation for neural-behavior alignment evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime


class SimulationDatasetGenerator:
    """
    Generate simulation-based datasets for evaluating neural-behavior alignment.
    Creates multimodal experimental setups with neural signals and behavioral data.
    """
    
    def __init__(self,
                 n_trials: int = 100,
                 n_subjects: int = 10,
                 n_conditions: int = 3,
                 sampling_rate: float = 1000.0,
                 trial_duration: float = 5.0):
        """
        Initialize dataset generator.
        
        Args:
            n_trials: Number of trials per subject/condition
            n_subjects: Number of subjects
            n_conditions: Number of experimental conditions
            sampling_rate: Sampling rate in Hz
            trial_duration: Duration of each trial in seconds
        """
        self.n_trials = n_trials
        self.n_subjects = n_subjects
        self.n_conditions = n_conditions
        self.sampling_rate = sampling_rate
        self.trial_duration = trial_duration
        self.n_samples = int(trial_duration * sampling_rate)
    
    def generate_neural_signal(self,
                              trial_type: str = 'decision',
                              noise_level: float = 0.1,
                              subject_id: int = 0) -> np.ndarray:
        """
        Generate synthetic neural signal.
        
        Args:
            trial_type: Type of trial ('decision', 'reward', 'action')
            noise_level: Noise level
            subject_id: Subject ID for individual variability
        
        Returns:
            Neural signal array
        """
        time = np.linspace(0, self.trial_duration, self.n_samples)
        
        # Base signal with subject-specific frequency
        base_freq = 10 + subject_id * 2  # Hz
        signal = np.sin(2 * np.pi * base_freq * time)
        
        # Add trial-type specific modulation
        if trial_type == 'decision':
            # Decision-related activity
            decision_time = self.trial_duration * 0.6
            decision_signal = np.exp(-((time - decision_time) ** 2) / 0.1)
            signal += 2 * decision_signal
        elif trial_type == 'reward':
            # Reward-related activity
            reward_time = self.trial_duration * 0.8
            reward_signal = np.exp(-((time - reward_time) ** 2) / 0.05)
            signal += 1.5 * reward_signal
        elif trial_type == 'action':
            # Action-related activity
            action_times = [self.trial_duration * 0.3, self.trial_duration * 0.7]
            for action_time in action_times:
                action_signal = np.exp(-((time - action_time) ** 2) / 0.02)
                signal += 1.0 * action_signal
        
        # Add noise
        noise = np.random.normal(0, noise_level, self.n_samples)
        signal += noise
        
        # Add slow drift
        drift = np.linspace(0, 0.1, self.n_samples)
        signal += drift
        
        return signal
    
    def generate_behavioral_data(self,
                                condition: int = 0,
                                subject_id: int = 0) -> Dict:
        """
        Generate behavioral data for a trial.
        
        Args:
            condition: Experimental condition
            subject_id: Subject ID
        
        Returns:
            Dictionary with behavioral data
        """
        # Generate action sequence
        n_actions = np.random.randint(3, 8)
        actions = np.random.randint(0, 4, n_actions)
        
        # Generate action times
        action_times = np.sort(np.random.uniform(0.2, self.trial_duration - 0.2, n_actions))
        
        # Generate rewards (condition-dependent)
        if condition == 0:
            reward_prob = 0.7
        elif condition == 1:
            reward_prob = 0.5
        else:
            reward_prob = 0.3
        
        rewards = (np.random.random(n_actions) < reward_prob).astype(float)
        rewards = rewards * np.random.uniform(0.5, 1.5, n_actions)
        
        # Generate reaction times (subject-dependent)
        base_rt = 0.3 + subject_id * 0.02
        reaction_times = np.random.normal(base_rt, 0.05, n_actions)
        reaction_times = np.clip(reaction_times, 0.1, 1.0)
        
        # Generate decision values
        decision_values = np.random.uniform(0, 1, n_actions)
        if condition == 0:
            decision_values += 0.2  # Higher values in condition 0
        
        return {
            'actions': actions,
            'action_times': action_times,
            'rewards': rewards,
            'reaction_times': reaction_times,
            'decision_values': decision_values,
            'total_reward': np.sum(rewards),
            'n_actions': n_actions
        }
    
    def generate_trial(self,
                      trial_id: int,
                      subject_id: int,
                      condition: int) -> Dict:
        """Generate complete trial data."""
        # Determine trial type
        trial_types = ['decision', 'reward', 'action']
        trial_type = trial_types[trial_id % len(trial_types)]
        
        # Generate neural signal
        neural_signal = self.generate_neural_signal(
            trial_type=trial_type,
            subject_id=subject_id
        )
        
        # Generate behavioral data
        behavioral_data = self.generate_behavioral_data(
            condition=condition,
            subject_id=subject_id
        )
        
        # Create time axis
        time_axis = np.linspace(0, self.trial_duration, self.n_samples)
        
        return {
            'trial_id': trial_id,
            'subject_id': subject_id,
            'condition': condition,
            'trial_type': trial_type,
            'neural_signal': neural_signal,
            'time_axis': time_axis,
            **behavioral_data
        }
    
    def generate_full_dataset(self, output_dir: str = 'data') -> pd.DataFrame:
        """
        Generate full dataset with all trials, subjects, and conditions.
        
        Args:
            output_dir: Directory to save dataset
        
        Returns:
            DataFrame with trial metadata
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_trials = []
        
        for subject_id in range(self.n_subjects):
            for condition in range(self.n_conditions):
                for trial_id in range(self.n_trials):
                    trial_data = self.generate_trial(trial_id, subject_id, condition)
                    
                    # Save individual trial data
                    trial_file = os.path.join(
                        output_dir,
                        f'subject_{subject_id}_condition_{condition}_trial_{trial_id}.npz'
                    )
                    np.savez_compressed(
                        trial_file,
                        neural_signal=trial_data['neural_signal'],
                        time_axis=trial_data['time_axis'],
                        actions=trial_data['actions'],
                        action_times=trial_data['action_times'],
                        rewards=trial_data['rewards'],
                        reaction_times=trial_data['reaction_times'],
                        decision_values=trial_data['decision_values']
                    )
                    
                    # Store metadata
                    all_trials.append({
                        'trial_id': trial_id,
                        'subject_id': subject_id,
                        'condition': condition,
                        'trial_type': trial_data['trial_type'],
                        'n_actions': trial_data['n_actions'],
                        'total_reward': trial_data['total_reward'],
                        'file_path': trial_file
                    })
        
        # Create DataFrame
        df = pd.DataFrame(all_trials)
        
        # Save metadata
        metadata_file = os.path.join(output_dir, 'dataset_metadata.csv')
        df.to_csv(metadata_file, index=False)
        
        # Save summary statistics
        summary = {
            'n_trials': self.n_trials,
            'n_subjects': self.n_subjects,
            'n_conditions': self.n_conditions,
            'sampling_rate': self.sampling_rate,
            'trial_duration': self.trial_duration,
            'generation_date': datetime.now().isoformat()
        }
        
        summary_file = os.path.join(output_dir, 'dataset_summary.txt')
        with open(summary_file, 'w') as f:
            for key, value in summary.items():
                f.write(f'{key}: {value}\n')
        
        return df
    
    def load_trial(self, file_path: str) -> Dict:
        """Load trial data from file."""
        data = np.load(file_path)
        return {
            'neural_signal': data['neural_signal'],
            'time_axis': data['time_axis'],
            'actions': data['actions'],
            'action_times': data['action_times'],
            'rewards': data['rewards'],
            'reaction_times': data['reaction_times'],
            'decision_values': data['decision_values']
        }

