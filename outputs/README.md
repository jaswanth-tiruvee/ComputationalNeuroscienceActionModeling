# Outputs Directory

This directory contains generated outputs from running the pipeline.

## Structure

- `models/` - Trained model checkpoints (`.pt` files)
- `figures/` - Generated visualizations (`.png` files)
- `dataset_summary.txt` - Statistical summary of generated datasets

## Example Outputs

Example visualizations are included in `figures/examples/` to demonstrate the types of outputs generated:

- `example_learning_curve.png` - Training progress over episodes
- `example_action_sequences.png` - Action sequences across trials
- `example_condition_comparison.png` - Statistical comparisons across conditions

## Generating Outputs

To generate outputs, run:

```bash
# Generate dataset
python main.py --mode dataset

# Train agent
python main.py --mode train

# Process signals and generate visualizations
python main.py --mode process

# Analyze action sequences
python main.py --mode analyze
```

All outputs will be saved here based on the configuration in `config.yaml`.

## Note

Most output files are excluded from git via `.gitignore` because they can be regenerated. Only example outputs in `figures/examples/` are included for demonstration purposes.

