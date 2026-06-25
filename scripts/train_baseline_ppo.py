from pathlib import Path

import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import FlattenObservation


ENV_ID = "MiniGrid-Empty-5x5-v0"
MODEL_PATH = Path("results/baseline_ppo")
TOTAL_TIMESTEPS = 50_000
SEED = 42


def make_environment() -> gym.Env:
    env = gym.make(ENV_ID)
    env = ImgObsWrapper(env)
    env = FlattenObservation(env)
    env = Monitor(env)
    return env


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    env = make_environment()

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=2.5e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        seed=SEED,
        verbose=1,
        tensorboard_log="runs/baseline_ppo",
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_PATH)

    env.close()

    print(f"Baseline PPO model saved to: {MODEL_PATH}.zip")


if __name__ == "__main__":
    main()