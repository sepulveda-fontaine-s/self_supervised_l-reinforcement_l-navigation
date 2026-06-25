# Self-Supervised Reinforcement Learning in MiniGrid

This project investigates whether self-supervised representation learning improves reinforcement learning performance and sample efficiency in a visual navigation task.

A convolutional autoencoder is trained on unlabeled MiniGrid observations. Its frozen encoder compresses each observation into a 16-dimensional latent representation, which is then used by a PPO agent.

## Results

| Agent        | Observation                              | Average reward | Success rate |
| ------------ | ---------------------------------------- | -------------: | -----------: |
| Baseline PPO | Flattened raw observation                |         0.9550 |         100% |
| SSL PPO      | 16-dimensional pretrained representation |         0.9460 |         100% |

Both agents reached a 100% success rate after 5,000 training timesteps. In this simple environment, self-supervised pretraining produced a compact and usable representation but did not improve final performance or sample efficiency.

![Sample-efficiency comparison](images/07_sample_efficiency_comparison.png)

## Project Structure

```text
ssl-rl-navigation/
├── data/
├── images/
├── results/
├── scripts/
├── src/
│   ├── models/
│   └── rl/
├── REPORT.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup

Create and activate a virtual environment on Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Project

Collect unlabeled observations:

```powershell
python -m scripts.collect_observations
```

Train the convolutional autoencoder:

```powershell
python -m scripts.train_autoencoder
```

Inspect the learned latent space:

```powershell
python -m scripts.inspect_latent_space
```

Train the baseline PPO agent:

```powershell
python -m scripts.train_baseline_ppo
```

Train PPO using the pretrained encoder:

```powershell
python -m scripts.train_ssl_ppo
```

Evaluate both agents:

```powershell
python -m scripts.evaluate_baseline_ppo
python -m scripts.evaluate_ssl_ppo
```

Run the sample-efficiency comparison:

```powershell
python -m scripts.compare_sample_efficiency
python -m scripts.plot_sample_efficiency
```

## Method

The workflow consists of four stages:

1. Collect 5,000 unlabeled observations using a random policy.
2. Train a convolutional autoencoder to reconstruct MiniGrid observations.
3. Freeze the 16-dimensional encoder and use it as an observation wrapper for PPO.
4. Compare the SSL-based agent with a PPO baseline trained on flattened raw observations.

MiniGrid observations use categorical channels for object type, colour, and state. Channel-specific normalization is therefore applied instead of conventional RGB normalization.

## Key Finding

The initial encoder failed because MiniGrid observations were incorrectly divided by 255, producing nearly constant latent representations and a 0% reinforcement learning success rate. After applying channel-specific normalization, the SSL-based PPO agent reached a 100% success rate.

This demonstrates that a low reconstruction loss does not guarantee that a learned representation is useful for downstream decision-making.

## Detailed Report

See [`REPORT.md`](REPORT.md) for the full methodology, experiments, figures, discussion, and conclusions.

## Technologies

* Python
* PyTorch
* Gymnasium
* MiniGrid
* Stable-Baselines3
* NumPy
* Matplotlib
* scikit-learn
