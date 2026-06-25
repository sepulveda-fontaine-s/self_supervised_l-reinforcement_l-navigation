from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.autoencoder import MiniGridAutoencoder
from src.models.preprocessing import normalize_minigrid

DATA_PATH = Path("data/minigrid_observations.npz")
MODEL_PATH = Path("results/autoencoder.pt")

EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
SEED = 42


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    images = np.load(DATA_PATH)["images"]
    images = torch.tensor(images, dtype=torch.float32)

    images = images.permute(0, 3, 1, 2)
    images = normalize_minigrid(images)

    dataset = TensorDataset(images)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = MiniGridAutoencoder(latent_dim=16).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for (batch,) in dataloader:
            batch = batch.to(device)

            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.size(0)

        average_loss = total_loss / len(dataset)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"- Loss: {average_loss:.6f}"
        )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()