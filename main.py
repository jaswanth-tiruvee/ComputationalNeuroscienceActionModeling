"""
Main entry point for Computational Neuroscience Action Modeling.
RL simulation and signal processing pipeline.
"""

import numpy as np
import yaml
import os
import argparse
from pathlib import Path
import torch
import random

from envs import BehavioralTaskEnv, MultiAgentBehavioralEnv
from models import DQNAgent, PPOAgent
from utils import (
    SimulationDatasetGenerator,
    EventTriggeredFeatureExtractor,
    SignalProcessor,
    BehavioralVisualizer,
    NeuralVisualizer,
    SummaryGenerator
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str = 'config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_environment(config):
    """Create environment based on configuration."""
    env_config = config['environment']
    
    if env_config['type'] == 'BehavioralTaskEnv':
        env = BehavioralTaskEnv(
            grid_size=env_config['grid_size'],
            n_goals=env_config['n_goals'],
            max_steps=env_config['max_steps'],
            reward_scale=env_config['reward_scale'],
            noise_level=env_config['noise_level']
        )
    elif env_config['type'] == 'MultiAgentBehavioralEnv':
        env = MultiAgentBehavioralEnv(
            n_agents=env_config['n_agents'],
            grid_size=env_config['grid_size'],
            n_goals=env_config['n_goals'],
            max_steps=env_config['max_steps']
        )
    else:
        raise ValueError(f"Unknown environment type: {env_config['type']}")
    
    return env


def create_agent(env, config):
    """Create agent based on configuration."""
    agent_config = config['agent']
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
    
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    if agent_config['type'] == 'DQN':
        dqn_config = agent_config['dqn']
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=dqn_config['lr'],
            gamma=dqn_config['gamma'],
            epsilon_start=dqn_config['epsilon_start'],
            epsilon_end=dqn_config['epsilon_end'],
            epsilon_decay=dqn_config['epsilon_decay'],
            batch_size=dqn_config['batch_size'],
            buffer_size=dqn_config['buffer_size'],
            target_update_freq=dqn_config['target_update_freq'],
            device=device
        )
    elif agent_config['type'] == 'PPO':
        ppo_config = agent_config['ppo']
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=ppo_config['lr'],
            gamma=ppo_config['gamma'],
            gae_lambda=ppo_config['gae_lambda'],
            clip_epsilon=ppo_config['clip_epsilon'],
            value_coef=ppo_config['value_coef'],
            entropy_coef=ppo_config['entropy_coef'],
            max_grad_norm=ppo_config['max_grad_norm'],
            device=device
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_config['type']}")
    
    return agent


def train_agent(env, agent, config):
    """Train the agent."""
    training_config = config['training']
    n_episodes = training_config['n_episodes']
    eval_frequency = training_config['eval_frequency']
    save_frequency = training_config['save_frequency']
    
    paths = config['paths']
    os.makedirs(paths['models_dir'], exist_ok=True)
    
    episode_rewards = []
    episode_lengths = []
    
    print(f"Starting training with {agent.__class__.__name__}...")
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # For PPO, collect trajectory
        if isinstance(agent, PPOAgent):
            states, actions, log_probs, rewards_list, values, dones = [], [], [], [], [], []
        
        while not done:
            if isinstance(agent, DQNAgent):
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.train_step()
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
            elif isinstance(agent, PPOAgent):
                action, log_prob, value = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards_list.append(reward)
                values.append(value)
                dones.append(done)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
        
        # Update PPO at end of episode
        if isinstance(agent, PPOAgent) and len(states) > 0:
            next_value = 0.0 if done else agent.network.critic(
                torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
            ).item()
            agent.update(states, actions, log_probs, rewards_list, values, dones, next_value)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if isinstance(agent, DQNAgent):
            agent.update_epsilon()
        
        # Evaluation and logging
        if (episode + 1) % eval_frequency == 0:
            avg_reward = np.mean(episode_rewards[-eval_frequency:])
            avg_length = np.mean(episode_lengths[-eval_frequency:])
            print(f"Episode {episode+1}/{n_episodes} - "
                  f"Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}")
        
        # Save model
        if (episode + 1) % save_frequency == 0:
            model_path = os.path.join(paths['models_dir'], 
                                     f"{config['experiment']['name']}_ep{episode+1}.pt")
            agent.save(model_path)
            print(f"Model saved to {model_path}")
    
    return episode_rewards, episode_lengths


def generate_dataset(config):
    """Generate simulation dataset."""
    print("Generating simulation dataset...")
    
    dataset_config = config['dataset']
    generator = SimulationDatasetGenerator(
        n_trials=dataset_config['n_trials'],
        n_subjects=dataset_config['n_subjects'],
        n_conditions=dataset_config['n_conditions'],
        sampling_rate=dataset_config['sampling_rate'],
        trial_duration=dataset_config['trial_duration']
    )
    
    df = generator.generate_full_dataset(output_dir=dataset_config['output_dir'])
    
    print(f"Dataset generated: {len(df)} trials")
    print(f"Metadata saved to {os.path.join(dataset_config['output_dir'], 'dataset_metadata.csv')}")
    
    return df, generator


def analyze_action_sequences(env, agent, config):
    """Analyze action sequences from trained agent."""
    print("Analyzing action sequences...")
    
    paths = config['paths']
    os.makedirs(paths['figures_dir'], exist_ok=True)
    
    visualizer = BehavioralVisualizer()
    
    # Generate multiple sequences
    sequences = []
    trajectories = []
    rewards = []
    
    for i in range(10):
        # Run episode manually to collect trajectory
        state = env.reset()
        episode_actions = []
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < 100:
            if isinstance(agent, DQNAgent):
                action = agent.select_action(state, training=False)
            else:
                action, _, _ = agent.select_action(state)
            
            next_state, reward, done, info = env.step(action)
            episode_actions.append(action)
            episode_reward += reward
            state = next_state
            step_count += 1
        
        sequences.append(np.array(episode_actions))
        rewards.append(episode_reward)
        
        # Get trajectory if available
        if hasattr(env, 'get_trajectory'):
            traj = env.get_trajectory()
            if traj and 'positions' in traj:
                trajectories.append(traj['positions'])
    
    # Visualize
    visualizer.plot_action_sequences(
        sequences,
        title="Action Sequences Across Trials",
        save_path=os.path.join(paths['figures_dir'], 'action_sequences.png')
    )
    
    if len(trajectories) > 0:
        visualizer.plot_trajectories(
            trajectories,
            title="Agent Trajectories",
            save_path=os.path.join(paths['figures_dir'], 'trajectories.png')
        )
    
    print(f"Visualizations saved to {paths['figures_dir']}")


def process_signals(config, generator):
    """Process signals and extract features."""
    print("Processing signals and extracting features...")
    
    paths = config['paths']
    os.makedirs(paths['figures_dir'], exist_ok=True)
    
    # Load metadata
    metadata_path = os.path.join(config['dataset']['output_dir'], 'dataset_metadata.csv')
    import pandas as pd
    df = pd.read_csv(metadata_path)
    
    # Initialize extractors
    signal_config = config['signal_processing']
    extractor = EventTriggeredFeatureExtractor(
        window_pre=signal_config['window_pre'],
        window_post=signal_config['window_post'],
        sampling_rate=signal_config['sampling_rate']
    )
    
    neural_viz = NeuralVisualizer()
    
    # Process a sample of trials
    sample_trials = df.sample(min(5, len(df)))
    
    for idx, row in sample_trials.iterrows():
        trial_data = generator.load_trial(row['file_path'])
        
        # Extract event-triggered average for actions
        eta_data = extractor.extract_event_triggered_average(
            trial_data['neural_signal'],
            trial_data['action_times'],
            time_axis=trial_data['time_axis']
        )
        
        # Visualize
        neural_viz.plot_event_triggered_average(
            eta_data,
            title=f"ETA - Subject {row['subject_id']}, Condition {row['condition']}",
            save_path=os.path.join(paths['figures_dir'], 
                                  f'eta_subj{row["subject_id"]}_cond{row["condition"]}_trial{row["trial_id"]}.png')
        )
        
        # Neural-behavioral alignment
        neural_viz.plot_neural_behavior_alignment(
            trial_data['neural_signal'],
            trial_data['action_times'],
            trial_data['actions'],
            trial_data['time_axis'],
            title=f"Alignment - Subject {row['subject_id']}, Condition {row['condition']}",
            save_path=os.path.join(paths['figures_dir'],
                                  f'alignment_subj{row["subject_id"]}_cond{row["condition"]}_trial{row["trial_id"]}.png')
        )
    
    print("Signal processing complete")


def generate_summaries(config, df):
    """Generate summary statistics and reports."""
    print("Generating summaries...")
    
    paths = config['paths']
    os.makedirs(paths['outputs_dir'], exist_ok=True)
    
    visualizer = BehavioralVisualizer()
    
    # Condition comparison
    if 'condition' in df.columns and 'total_reward' in df.columns:
        visualizer.plot_condition_comparison(
            df,
            value_col='total_reward',
            condition_col='condition',
            title="Reward Across Conditions",
            save_path=os.path.join(paths['figures_dir'], 'condition_comparison.png')
        )
    
    # Subject summary
    if 'subject_id' in df.columns and 'total_reward' in df.columns:
        visualizer.plot_subject_summary(
            df,
            subject_col='subject_id',
            value_col='total_reward',
            title="Subject Performance",
            save_path=os.path.join(paths['figures_dir'], 'subject_summary.png')
        )
    
    # Text summary
    summary = SummaryGenerator.generate_dataset_summary(
        df,
        output_path=os.path.join(paths['outputs_dir'], 'dataset_summary.txt')
    )
    print(summary)


def main():
    """Main pipeline."""
    parser = argparse.ArgumentParser(description='Computational Neuroscience Action Modeling')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'train', 'dataset', 'analyze', 'process'],
                       help='Execution mode')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['experiment']['seed'])
    
    # Create output directories
    paths = config['paths']
    for dir_path in paths.values():
        os.makedirs(dir_path, exist_ok=True)
    
    if args.mode in ['full', 'dataset']:
        # Generate dataset
        df, generator = generate_dataset(config)
        
        if args.mode == 'full':
            # Process signals
            process_signals(config, generator)
            
            # Generate summaries
            generate_summaries(config, df)
    
    if args.mode in ['full', 'train', 'analyze']:
        # Create environment
        env = create_environment(config)
        
        # Create agent
        agent = create_agent(env, config)
        
        if args.mode in ['full', 'train']:
            # Train agent
            episode_rewards, episode_lengths = train_agent(env, agent, config)
            
            # Plot learning curves
            visualizer = BehavioralVisualizer()
            visualizer.plot_reward_curves(
                [np.array(episode_rewards)],
                labels=[f"{agent.__class__.__name__}"],
                title="Training Progress",
                save_path=os.path.join(paths['figures_dir'], 'learning_curve.png')
            )
        
        if args.mode in ['full', 'analyze']:
            # Analyze action sequences
            analyze_action_sequences(env, agent, config)
    
    if args.mode == 'process':
        # Process existing dataset
        dataset_config = config['dataset']
        generator = SimulationDatasetGenerator(
            n_trials=dataset_config['n_trials'],
            n_subjects=dataset_config['n_subjects'],
            n_conditions=dataset_config['n_conditions'],
            sampling_rate=dataset_config['sampling_rate'],
            trial_duration=dataset_config['trial_duration']
        )
        metadata_path = os.path.join(config['dataset']['output_dir'], 'dataset_metadata.csv')
        import pandas as pd
        df = pd.read_csv(metadata_path)
        process_signals(config, generator)
        generate_summaries(config, df)
    
    print("Pipeline complete!")


if __name__ == '__main__':
    main()
