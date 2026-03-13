from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def create_sliding_windows(data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Split a (T, H, W) sequence into temporal windows."""
    if data.shape[0] < window_size:
        return np.empty((0, window_size, *data.shape[1:]), dtype=data.dtype)

    windows = []
    for start in range(0, data.shape[0] - window_size + 1, stride):
        windows.append(data[start : start + window_size])
    return np.asarray(windows)


@dataclass(frozen=True)
class SampleInfo:
    track_dir: str
    label: int


class FallDetectionDataset(Dataset):
    """Video-level dataset backed by saved per-frame height maps."""

    def __init__(
        self,
        root_dir: str,
        window_size: int = 32,
        stride: int = 16,
        is_train: bool = True,
        positive_dir: str = "Fall",
        negative_dir: str = "NonFall",
        noise_std: float = 0.01,
        horizontal_flip_prob: float = 0.5,
    ) -> None:
        self.root_dir = root_dir
        self.window_size = window_size
        self.stride = stride
        self.is_train = is_train
        self.noise_std = noise_std
        self.horizontal_flip_prob = horizontal_flip_prob
        self.samples: list[SampleInfo] = []

        self._load_category(positive_dir, label=1)
        self._load_category(negative_dir, label=0)

        if not self.samples:
            raise FileNotFoundError(
                f"No track directories were found under {root_dir!r}. "
                f"Expected {positive_dir}/<video>/track_* and {negative_dir}/<video>/track_*."
            )

    def _load_category(self, category_dir: str, label: int) -> None:
        pattern = os.path.join(self.root_dir, category_dir, "*", "track_*")
        for track_dir in tqdm(sorted(glob.glob(pattern)), desc=f"Loading {category_dir}"):
            npy_files = glob.glob(os.path.join(track_dir, "*.npy"))
            if npy_files:
                self.samples.append(SampleInfo(track_dir=track_dir, label=label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        npy_files = sorted(glob.glob(os.path.join(sample.track_dir, "*.npy")))
        height_maps = [np.load(file_path).astype(np.float32) for file_path in npy_files]
        sequence = np.stack(height_maps, axis=0)

        if self.is_train:
            if random.random() < self.horizontal_flip_prob:
                sequence = np.flip(sequence, axis=2).copy()
            if self.noise_std > 0:
                sequence = sequence + np.random.normal(0.0, self.noise_std, sequence.shape)

        windows = create_sliding_windows(sequence, self.window_size, self.stride)
        return torch.from_numpy(windows).float(), torch.tensor(sample.label, dtype=torch.float32)


def custom_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep a video's variable number of windows intact."""
    windows, label = batch[0]
    return windows, label

