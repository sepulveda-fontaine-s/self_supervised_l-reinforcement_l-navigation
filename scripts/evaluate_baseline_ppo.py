from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import minigrid
import numpy as np
from gymnasium.wrappers import FlattenObservation
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO


ENV_ID = "MiniGrid-Empty-5x5-v0"
MODEL_PATH = Path("results/baseline_ppo.zip")
OUTPUT_PATH = Path("images/04_baseline_episode_rewards.png")

NUM_EPISODES = 100
SEED = 42


def make_environment() -> gym.Env:
    env = gym.make(ENV_ID)
    env = ImgObsWrapper(env)
    env = FlattenObservation(env)
    return env


def main() -> None:
    env = make_environment()
    model = PPO.load(MODEL_PATH)

    episode_rewards: list[float] = []
    successes = 0

    for episode in range(NUM_EPISODES):
        observation, _ = env.reset(seed=SEED + episode)
        terminated = False
        truncated = False
        total_reward = 0.0

        while not terminated and not truncated:
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

        episode_rewards.append(total_reward)

        if total_reward > 0:
            successes += 1

    env.close()

    average_reward = float(np.mean(episode_rewards))
    success_rate = successes / NUM_EPISODES

    print(f"Average reward: {average_reward:.4f}")
    print(f"Success rate: {success_rate:.2%}")

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(range(1, NUM_EPISODES + 1), episode_rewards)
    axis.set_title("Baseline PPO Evaluation")
    axis.set_xlabel("Evaluation episode")
    axis.set_ylabel("Episode reward")
    axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=200)
    plt.close(figure)

    print(f"Saved evaluation plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()