"""
Sampler design for Two-Stage Cluster-Balanced Epoch Sampling.
Ensures images with multiple alternative annotations receive identical visual exposure frequency
while utilizing all valid ground-truth annotations as natural stochastic label regularization.
"""

import random
from typing import List, Dict, Any, Iterator

import torch
from torch.utils.data import Sampler


class EpochShuffleSampler(Sampler[int]):
    """
    Deterministic epoch-aware index sampler for the production Stage 4
    segmentation datasets.

    The production datasets already contain exactly one primary record per
    unique image-content cluster, so no alternate-annotation selection is
    possible or permitted. Each epoch returns every index exactly once in a
    seed/epoch-derived permutation.

    Reproducibility contract: ``list(sampler)`` is identical for the same
    ``(seed, epoch)`` and changes across epochs.
    """

    def __init__(self, num_samples: int, seed: int = 42):
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative")
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.num_samples, generator=generator).tolist())

    def __len__(self) -> int:
        return self.num_samples


class TwoStageClusterSampler:
    """
    Legacy two-stage cluster sampler retained only for historical tests.

    It is NOT used by the production Stage 4 loaders. The approved policy is
    exactly one primary annotation per image-content cluster, enforced by
    ``DFUCanonicalDataset`` and ``EpochShuffleSampler``. This legacy class must
    not be wired into a loader, because its second stage can choose among
    multiple annotations.
    """
    def __init__(self, cluster_to_indices: Dict[str, List[int]], seed: int = 42):
        self.cluster_to_indices = cluster_to_indices
        self.unique_clusters = sorted(list(cluster_to_indices.keys()))
        self.num_unique_images = len(self.unique_clusters)
        self.rng = random.Random(seed)

    def set_epoch(self, epoch: int):
        """Sets random seed deterministically per epoch to guarantee reproducibility."""
        self.rng.seed(42 + epoch * 1000)

    def __iter__(self) -> Iterator[int]:
        # Shuffle clusters uniformly
        shuffled_clusters = list(self.unique_clusters)
        self.rng.shuffle(shuffled_clusters)

        # For each cluster, pick exactly one mask index
        epoch_indices = []
        for cluster_id in shuffled_clusters:
            candidates = self.cluster_to_indices[cluster_id]
            chosen_idx = self.rng.choice(candidates)
            epoch_indices.append(chosen_idx)

        return iter(epoch_indices)

    def __len__(self) -> int:
        return self.num_unique_images

