from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.autoencoder import MiniGridAutoencoder
from src.models.preprocessing import normalize_minigrid

DATA_PATH = Path("data/minigrid_observations.npz")
MODEL_PATH = Path("results/autoencoder.pt")
OUTPUT_PATH = Path("images/03_autoencoder_reconstructions.png")

NUM_SAMPLES = 6
SEED = 42


def main() -> None:
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = np.load(DATA_PATH)["images"]
    sample_images = images[:NUM_SAMPLES]

    input_tensor = torch.tensor(
        sample_images,
        dtype=torch.float32,
    ).permute(0, 3, 1, 2)

    input_tensor = normalize_minigrid(input_tensor)

    model = MiniGridAutoencoder(latent_dim=16).to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )
    model.eval()

    with torch.no_grad():
        reconstructions = model(input_tensor.to(device)).cpu()

    originals = input_tensor.permute(0, 2, 3, 1).numpy()
    reconstructed = reconstructions.permute(0, 2, 3, 1).numpy()

    figure, axes = plt.subplots(2, NUM_SAMPLES, figsize=(12, 4))

    for index in range(NUM_SAMPLES):
        axes[0, index].imshow(originals[index])
        axes[0, index].axis("off")

        axes[1, index].imshow(reconstructed[index])
        axes[1, index].axis("off")

    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstructed")

    figure.suptitle("Autoencoder Reconstruction Results")
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=200)
    plt.close(figure)

    print(f"Saved reconstruction image to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()