from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from src.models.autoencoder import MiniGridAutoencoder
from src.models.preprocessing import normalize_minigrid


class EncoderObservationWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        model_path: Path,
        latent_dim: int = 64,
    ) -> None:
        super().__init__(env)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.encoder = MiniGridAutoencoder(
            latent_dim=latent_dim
        ).to(self.device)

        self.encoder.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )

        self.encoder.eval()

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(latent_dim,),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        )

        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = normalize_minigrid(tensor)

        with torch.no_grad():
            latent = self.encoder.encode(tensor)

        return latent.squeeze(0).cpu().numpy().astype(np.float32)