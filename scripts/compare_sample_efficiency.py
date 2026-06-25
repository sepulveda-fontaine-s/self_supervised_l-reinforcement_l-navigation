from pathlib import Path

import gymnasium as gym
import minigrid
import numpy as np
from gymnasium.wrappers import FlattenObservation
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO

from src.rl.encoder_wrapper import EncoderObservationWrapper


ENV_ID = "MiniGrid-Empty-5x5-v0"
ENCODER_PATH = Path("results/autoencoder.pt")

TRAINING_STEPS = [1_000, 2_500, 5_000, 10_000, 25_000]
NUM_EVALUATION_EPISODES = 50
SEED = 42


def make_baseline_environment() -> gym.Env:
    env = gym.make(ENV_ID)
    env = ImgObsWrapper(env)
    env = FlattenObservation(env)
    return env


def make_ssl_environment() -> gym.Env:
    env = gym.make(ENV_ID)
    env = ImgObsWrapper(env)
    env = EncoderObservationWrapper(
        env=env,
        model_path=ENCODER_PATH,
        latent_dim=16,
    )
    return env


def evaluate(
    model: PPO,
    environment_factory,
) -> tuple[float, float]:
    env = environment_factory()

    rewards = []
    successes = 0

    for episode in range(NUM_EVALUATION_EPISODES):
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

        rewards.append(total_reward)

        if total_reward > 0:
            successes += 1

    env.close()

    return (
        float(np.mean(rewards)),
        successes / NUM_EVALUATION_EPISODES,
    )


def train_and_evaluate(
    name: str,
    environment_factory,
) -> list[tuple[int, float, float]]:
    results = []

    for steps in TRAINING_STEPS:
        env = environment_factory()

        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=2.5e-4,
            n_steps=512,
            batch_size=64,
            gamma=0.99,
            seed=SEED,
            verbose=0,
        )

        model.learn(total_timesteps=steps)
        env.close()

        average_reward, success_rate = evaluate(
            model,
            environment_factory,
        )

        results.append(
            (steps, average_reward, success_rate)
        )

        print(
            f"{name} | Steps: {steps:>6} | "
            f"Reward: {average_reward:.4f} | "
            f"Success: {success_rate:.2%}"
        )

    return results


def main() -> None:
    print("\nBaseline PPO")
    baseline_results = train_and_evaluate(
        "Baseline",
        make_baseline_environment,
    )

    print("\nSSL PPO")
    ssl_results = train_and_evaluate(
        "SSL",
        make_ssl_environment,
    )

    output_path = Path("results/sample_efficiency.npz")

    np.savez(
        output_path,
        baseline=np.asarray(baseline_results),
        ssl=np.asarray(ssl_results),
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()