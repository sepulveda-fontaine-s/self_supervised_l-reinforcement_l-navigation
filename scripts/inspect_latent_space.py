from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from src.models.autoencoder import MiniGridAutoencoder
from src.models.preprocessing import normalize_minigrid


DATA_PATH = Path("data/minigrid_observations.npz")
MODEL_PATH = Path("results/autoencoder.pt")
OUTPUT_PATH = Path("images/06_latent_space_pca.png")

NUM_SAMPLES = 1000
LATENT_DIM = 16
SEED = 42


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = np.load(DATA_PATH)["images"][:NUM_SAMPLES]

    inputs = torch.tensor(
        images,
        dtype=torch.float32,
    ).permute(0, 3, 1, 2)

    inputs = normalize_minigrid(inputs)

    model = MiniGridAutoencoder(latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )
    model.eval()

    with torch.no_grad():
        latent_vectors = model.encode(inputs.to(device)).cpu().numpy()

    print(f"Latent shape: {latent_vectors.shape}")
    print(f"Mean latent standard deviation: {latent_vectors.std(axis=0).mean():.6f}")
    print(f"Overall latent standard deviation: {latent_vectors.std():.6f}")

    projection = PCA(n_components=2).fit_transform(latent_vectors)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(projection[:, 0], projection[:, 1], s=10, alpha=0.5)
    axis.set_title("PCA Projection of Autoencoder Latent Representations")
    axis.set_xlabel("Principal component 1")
    axis.set_ylabel("Principal component 2")
    axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=200)
    plt.close(figure)

    print(f"Saved latent-space plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()