from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_PATH = Path("results/sample_efficiency.npz")
OUTPUT_PATH = Path("images/07_sample_efficiency_comparison.png")


def main() -> None:
    results = np.load(DATA_PATH)

    baseline = results["baseline"]
    ssl = results["ssl"]

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.plot(
        baseline[:, 0],
        baseline[:, 2] * 100,
        marker="o",
        label="Baseline PPO",
    )

    axis.plot(
        ssl[:, 0],
        ssl[:, 2] * 100,
        marker="o",
        label="SSL PPO",
    )

    axis.set_title("Sample Efficiency Comparison")
    axis.set_xlabel("Training timesteps")
    axis.set_ylabel("Success rate (%)")
    axis.set_ylim(-5, 105)
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=200)
    plt.close(figure)

    print(f"Saved comparison plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()