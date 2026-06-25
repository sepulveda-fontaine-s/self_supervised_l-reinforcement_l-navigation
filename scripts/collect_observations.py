from pathlib import Path

import gymnasium as gym
import minigrid
import numpy as np


ENV_ID = "MiniGrid-Empty-5x5-v0"
NUM_OBSERVATIONS = 5_000
OUTPUT_PATH = Path("data/minigrid_observations.npz")
SEED = 42


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(ENV_ID)
    observation, _ = env.reset(seed=SEED)

    images: list[np.ndarray] = []

    for _ in range(NUM_OBSERVATIONS):
        images.append(observation["image"])

        action = env.action_space.sample()
        observation, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            observation, _ = env.reset()

    image_array = np.asarray(images, dtype=np.uint8)

    np.savez_compressed(
        OUTPUT_PATH,
        images=image_array,
    )

    env.close()

    print(f"Saved {len(image_array)} observations")
    print(f"Dataset shape: {image_array.shape}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()