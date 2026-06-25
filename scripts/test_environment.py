import gymnasium as gym
import minigrid
from PIL import Image


def main() -> None:
    env = gym.make(
        "MiniGrid-Empty-5x5-v0",
        render_mode="rgb_array",
    )

    observation, info = env.reset(seed=42)

    frame = env.render()
    Image.fromarray(frame).save("images/01_minigrid_environment.png")

    print("Environment image saved successfully")

    env.close()


if __name__ == "__main__":
    main()