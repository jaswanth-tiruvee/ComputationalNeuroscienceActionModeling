# Computational Neuroscience Action Modeling

A comprehensive Python framework for modeling goal-directed behavioral patterns using reinforcement learning, with integrated signal processing tools for neural-behavioral alignment analysis.

## Overview

This project provides:

- **RL Models**: Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) agents for capturing action sequences across agents
- **Signal Processing**: Event-triggered feature extraction supporting representation learning for decision signals
- **Simulation Datasets**: Generation of multimodal experimental datasets for evaluating neural-behavior alignment
- **Visualization Tools**: Comprehensive plotting and summary generation for interpreting patterns across trials, subjects, and conditions

## Features

### 1. Reinforcement Learning Agents
- **DQN Agent**: Deep Q-Network with experience replay for stable learning
- **PPO Agent**: Proximal Policy Optimization for policy gradient learning
- Both agents support action sequence modeling and trajectory analysis

### 2. Custom Environments
- **BehavioralTaskEnv**: Single-agent goal-directed decision-making task
- **MultiAgentBehavioralEnv**: Multi-agent environment for analyzing action sequences across agents

### 3. Signal Processing
- **Event-Triggered Feature Extraction**: Align neural signals to behavioral events
- **Signal Processing Utilities**: Bandpass filtering, power spectrum analysis, phase locking
- **Decision Signal Extraction**: Identify and analyze decision-related neural activity

### 4. Dataset Generation
- **Simulation-Based Datasets**: Generate synthetic neural and behavioral data
- **Multi-Modal Setup**: Supports multiple subjects, conditions, and trial types
- **Neural-Behavioral Alignment**: Pre-aligned signals for evaluation

### 5. Visualization & Analysis
- **Behavioral Visualizations**: Action sequences, trajectories, learning curves
- **Neural Visualizations**: Event-triggered averages, neural-behavioral alignment
- **Summary Statistics**: Comprehensive reports across trials, subjects, and conditions

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ComputationalNeuroscienceActionModeling
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Training

Train an RL agent on the behavioral task:

```bash
python main.py --mode train
```

**Example Output:**
```
Starting training with DQNAgent...
Episode 25/200 - Avg Reward: -10.49, Avg Length: 37.0
Episode 50/200 - Avg Reward: -5.16, Avg Length: 31.4
Episode 100/200 - Avg Reward: -0.55, Avg Length: 26.2
Episode 200/200 - Avg Reward: 3.53, Avg Length: 17.4
Model saved to outputs/models/test_experiment_ep200.pt
```

### Generate Dataset

Generate a simulation dataset:

```bash
python main.py --mode dataset
```

**Example Output:**
```
Generating simulation dataset...
Dataset generated: 300 trials
Metadata saved to data/dataset_metadata.csv
```

### Full Pipeline

Run the complete pipeline (dataset generation, training, analysis):

```bash
python main.py --mode full
```

This runs all steps sequentially: dataset generation → training → signal processing → analysis.

### Process Existing Data

Process signals from existing dataset:

```bash
python main.py --mode process
```

**Example Output:**
```
Processing signals and extracting features...
Signal processing complete
Generating summaries...
Dataset Summary
==================================================
Total Trials: 300
Number of Subjects: 5
Number of Conditions: 3
Reward Statistics:
  Mean: 2.36
  Std: 1.69
Reward by Condition:
  Condition 0: 3.63 ± 1.48
  Condition 1: 2.32 ± 1.49
  Condition 2: 1.14 ± 1.02
```

## Configuration

The project includes two configuration files:
- `config.yaml` - Full configuration for production use
- `config_test.yaml` - Quick test configuration with reduced parameters for faster execution

Edit `config.yaml` (or create your own) to customize:

- **Environment**: Grid size, number of goals, reward scaling
- **Agent**: DQN or PPO parameters, learning rates, network architecture
- **Training**: Number of episodes, evaluation frequency
- **Signal Processing**: Time windows, sampling rates, filter parameters
- **Dataset**: Number of trials, subjects, conditions

Example configuration:

```yaml
environment:
  type: "BehavioralTaskEnv"
  grid_size: 10
  n_goals: 2
  max_steps: 100

agent:
  type: "DQN"
  dqn:
    lr: 0.001
    gamma: 0.99
    epsilon_start: 1.0
    epsilon_end: 0.01
```

## Usage Examples

### Training an Agent

```python
from envs import BehavioralTaskEnv
from models import DQNAgent
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create environment and agent
env = BehavioralTaskEnv(grid_size=10, n_goals=2)
agent = DQNAgent(state_dim=env.observation_space.shape[0],
                 action_dim=env.action_space.n)

# Training loop
for episode in range(1000):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state, training=True)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.train_step()
        state = next_state
```

### Signal Processing

```python
from utils import EventTriggeredFeatureExtractor

# Initialize extractor
extractor = EventTriggeredFeatureExtractor(
    window_pre=0.5,
    window_post=1.0,
    sampling_rate=1000.0
)

# Extract event-triggered average
eta_data = extractor.extract_event_triggered_average(
    neural_signal,
    action_times,
    time_axis=time_axis
)

# Access results
eta = eta_data['eta']
time_axis_eta = eta_data['time_axis']
n_events = eta_data['n_events']
```

### Dataset Generation

```python
from utils import SimulationDatasetGenerator

# Create generator
generator = SimulationDatasetGenerator(
    n_trials=100,
    n_subjects=10,
    n_conditions=3,
    sampling_rate=1000.0,
    trial_duration=5.0
)

# Generate dataset
df = generator.generate_full_dataset(output_dir='data')

# Load a trial
trial_data = generator.load_trial('data/subject_0_condition_0_trial_0.npz')
```

### Visualization

```python
from utils import BehavioralVisualizer, NeuralVisualizer

# Behavioral visualization
behavioral_viz = BehavioralVisualizer()
behavioral_viz.plot_action_sequences(
    action_sequences,
    labels=['Agent 1', 'Agent 2'],
    save_path='outputs/action_sequences.png'
)

# Neural visualization
neural_viz = NeuralVisualizer()
neural_viz.plot_event_triggered_average(
    eta_data,
    save_path='outputs/eta.png'
)
```

## Project Structure

```
ComputationalNeuroscienceActionModeling/
├── main.py                 # Main entry point
├── config.yaml             # Configuration file
├── requirements.txt        # Dependencies
├── README.md              # This file
├── envs/                  # Custom environments
│   ├── __init__.py
│   └── behavioral_task.py
├── models/                # RL agents
│   ├── __init__.py
│   ├── dqn_agent.py
│   └── ppo_agent.py
├── utils/                 # Utilities
│   ├── __init__.py
│   ├── signal_processing.py
│   ├── dataset_generator.py
│   └── visualization.py
├── data/                  # Generated datasets
├── outputs/               # Outputs
│   ├── models/           # Saved models
│   └── figures/          # Visualizations
└── notebooks/            # Jupyter notebooks (optional)
```

## Outputs

The pipeline generates comprehensive outputs for analysis and visualization:

### Generated Files

- **Models**: Trained RL agents saved in `outputs/models/`
  - Model checkpoints at specified intervals (e.g., `test_experiment_ep100.pt`)
  - Contains network weights, optimizer state, and training parameters
  
- **Figures**: Visualizations in `outputs/figures/`
  - `action_sequences.png` - Action sequences across multiple trials
  - `trajectories.png` - 2D agent trajectories with start/end markers and goals
  - `learning_curve.png` - Training progress showing reward over episodes
  - `condition_comparison.png` - Box and violin plots comparing performance across conditions
  - `subject_summary.png` - Bar chart of mean performance per subject
  - `eta_*.png` - Event-triggered averages for neural signals aligned to actions
  - `alignment_*.png` - Neural-behavioral alignment plots showing signals with action markers

- **Data**: Generated datasets in `data/`
  - Individual trial files: `subject_{id}_condition_{id}_trial_{id}.npz`
  - `dataset_metadata.csv` - Trial metadata with subject, condition, and performance metrics
  - `dataset_summary.txt` - Statistical summary of the dataset

### Example Outputs

#### Training Results
```
Starting training with DQNAgent...
Episode 25/200 - Avg Reward: -10.49, Avg Length: 37.0
Episode 50/200 - Avg Reward: -5.16, Avg Length: 31.4
Episode 100/200 - Avg Reward: -0.55, Avg Length: 26.2
Episode 150/200 - Avg Reward: -0.23, Avg Length: 24.3
Episode 200/200 - Avg Reward: 3.53, Avg Length: 17.4
```

The agent shows clear learning progress, with rewards increasing from negative values to positive, and episode lengths decreasing as the agent learns more efficient paths to goals.

#### Dataset Summary
```
Dataset Summary
==================================================
Total Trials: 300
Number of Subjects: 5
Number of Conditions: 3

Reward Statistics:
  Mean: 2.36
  Std: 1.69
  Min: 0.00
  Max: 7.18

Reward by Condition:
  Condition 0: 3.63 ± 1.48
  Condition 1: 2.32 ± 1.49
  Condition 2: 1.14 ± 1.02
```

This shows clear condition-dependent differences in reward, with Condition 0 having the highest mean reward and Condition 2 the lowest.

#### Generated Visualizations

1. **Action Sequences** (`action_sequences.png`)
   - Shows action choices over time for multiple trials
   - Helps identify patterns and strategies in agent behavior
   - Useful for comparing different agents or training stages

2. **Agent Trajectories** (`trajectories.png`)
   - 2D visualization of agent paths through the environment
   - Start positions (squares), end positions (stars), and goals (X markers)
   - Reveals spatial navigation strategies

3. **Learning Curves** (`learning_curve.png`)
   - Smoothed reward curves showing training progress
   - Demonstrates convergence and learning stability
   - Essential for hyperparameter tuning

4. **Event-Triggered Averages** (`eta_*.png`)
   - Neural signals aligned to behavioral events (actions)
   - Shows average neural response around action times
   - Includes standard deviation bands for uncertainty
   - Critical for neural-behavioral alignment analysis

5. **Neural-Behavioral Alignment** (`alignment_*.png`)
   - Dual-panel plots showing:
     - Top: Neural signal over time with action markers
     - Bottom: Action timeline with color-coded actions
   - Enables direct visualization of signal-behavior relationships

6. **Condition Comparison** (`condition_comparison.png`)
   - Statistical comparison across experimental conditions
   - Box plots show distributions
   - Violin plots show density estimates
   - Identifies significant condition effects

7. **Subject Summary** (`subject_summary.png`)
   - Bar chart with error bars showing mean performance per subject
   - Highlights inter-subject variability
   - Useful for identifying outlier subjects

### Running the Full Pipeline

To generate all outputs:

```bash
# Generate dataset
python main.py --mode dataset

# Train agent
python main.py --mode train

# Process signals and generate visualizations
python main.py --mode process

# Analyze action sequences
python main.py --mode analyze

# Or run everything at once
python main.py --mode full
```

All outputs will be saved in the `outputs/` and `data/` directories as specified in `config.yaml`.

### Output File Structure

After running the pipeline, you'll find:

```
outputs/
├── models/
│   ├── test_experiment_ep50.pt      # Model checkpoint at episode 50
│   ├── test_experiment_ep100.pt    # Model checkpoint at episode 100
│   ├── test_experiment_ep150.pt    # Model checkpoint at episode 150
│   └── test_experiment_ep200.pt    # Final model checkpoint
├── figures/
│   ├── action_sequences.png         # ~288 KB - Action sequences across trials
│   ├── trajectories.png             # Agent path visualizations
│   ├── learning_curve.png           # Training progress
│   ├── condition_comparison.png     # Statistical comparisons
│   ├── subject_summary.png          # Subject performance
│   ├── eta_*.png                    # Event-triggered averages (~400 KB each)
│   └── alignment_*.png               # Neural-behavioral alignment (~400 KB each)
└── dataset_summary.txt              # Text summary of dataset statistics

data/
├── dataset_metadata.csv             # Trial metadata (300 rows for test config)
├── dataset_summary.txt              # Dataset generation summary
└── subject_*_condition_*_trial_*.npz # Individual trial data files
```

**Note**: File sizes and counts depend on configuration parameters. The test configuration generates 300 trials (5 subjects × 3 conditions × 20 trials).

**Example Run Results** (using `config_test.yaml`):
- ✅ 4 trained model checkpoints saved
- ✅ 15 visualization figures generated (~6 MB total)
- ✅ 300 trial data files created (~19 MB total)
- ✅ Complete dataset metadata and summaries
- ✅ All visualizations successfully generated and saved

The pipeline successfully demonstrates:
- RL agent learning (rewards improved from -10.49 to 3.53)
- Dataset generation with multiple subjects and conditions
- Signal processing and event-triggered feature extraction
- Comprehensive visualization generation
- Statistical analysis across conditions and subjects

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Gym
- PyYAML
- SciPy

## Citation

If you use this code in your research, please cite:

```
Computational Neuroscience Action Modeling – Python, RL, Signal Processing 2023
```

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on the repository.
