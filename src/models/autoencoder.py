import torch
from torch import nn


class MiniGridAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.encoder_linear = nn.Linear(32 * 7 * 7, latent_dim)

        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.ReLU(),
        )

        self.decoder_cnn = nn.Sequential(
            nn.Unflatten(1, (32, 7, 7)),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder_cnn(x)
        return self.encoder_linear(features)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        features = self.decoder_linear(z)
        return self.decoder_cnn(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return self.decode(latent)