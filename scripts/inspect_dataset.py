from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_PATH = Path("data/minigrid_observations.npz")
OUTPUT_PATH = Path("images/02_dataset_samples.png")
NUM_SAMPLES = 8


def main() -> None:
    dataset = np.load(DATA_PATH)
    images = dataset["images"]

    indices = np.linspace(
        0,
        len(images) - 1,
        NUM_SAMPLES,
        dtype=int,
    )

    figure, axes = plt.subplots(2, 4, figsize=(10, 5))

    for axis, index in zip(axes.flat, indices):
        axis.imshow(images[index])
        axis.set_title(f"Sample {index}")
        axis.axis("off")

    figure.suptitle("Collected MiniGrid Observations")
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=200)
    plt.close(figure)

    print(f"Dataset shape: {images.shape}")
    print(f"Saved visualization to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()