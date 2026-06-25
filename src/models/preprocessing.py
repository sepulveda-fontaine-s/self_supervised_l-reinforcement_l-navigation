import torch


CHANNEL_MAX_VALUES = torch.tensor(
    [10.0, 5.0, 2.0],
).view(1, 3, 1, 1)


def normalize_minigrid(images: torch.Tensor) -> torch.Tensor:
    maximums = CHANNEL_MAX_VALUES.to(images.device)
    return images / maximums