"""Experiment configuration."""

from dataclasses import asdict, dataclass, field
from typing import Literal

import yaml


@dataclass(frozen=True)
class DataConfig:
    """Data-generating distribution and dataset sizing."""

    dist: Literal[
        'gaussian',
        'low_rank_gaussian',
        'anisotropic_gaussian',
        'uniform',
        'laplace',
        'mixture',
        'swiss_roll',
        's_curve',
    ] = 'gaussian'
    d_in: int = 32
    n_samples: int = 4096
    mean: float = 0.0
    std: float = 1.0
    intrinsic_dim: int = 4
    noise_std: float = 0.1
    decay: float = 0.8
    n_components: int = 4
    splits: tuple[float, float, float] = (0.8, 0.1, 0.1)
    standardize: bool = False
    seed: int = 0

    def __post_init__(self):
        object.__setattr__(self, 'splits', tuple(self.splits))


@dataclass(frozen=True)
class ModelConfig:
    """Autoencoder architecture. d_mid=None gives a single linear layer."""

    kind: Literal['ae', 'vae'] = 'ae'
    d_in: int = 32
    d_hidden: int = 4
    d_mid: int | None = None
    activation: Literal['identity', 'relu', 'leaky_relu'] = 'identity'
    beta: float = 1.0
    recon: Literal['mse', 'gaussian_nll'] = 'mse'


@dataclass(frozen=True)
class TrainConfig:
    """Optimization and checkpointing."""

    optimizer: Literal['adam', 'sgd'] = 'adam'
    lr: float = 1e-2
    batch_size: int = 256
    steps: int = 2000
    seed: int = 0
    eval_every: int = 200
    run_dir: str = 'runs'


@dataclass(frozen=True)
class Config:
    """Top-level config."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        if self.data.d_in != self.model.d_in:
            raise ValueError(
                f'd_in mismatch: data={self.data.d_in}, model={self.model.d_in}'
            )


def load(path: str) -> Config:
    """Builds a Config from a YAML file (omitted fields use defaults)."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return Config(
        data=DataConfig(**raw.get('data', {})),
        model=ModelConfig(**raw.get('model', {})),
        train=TrainConfig(**raw.get('train', {})),
    )


def dump(cfg: Config, path: str) -> None:
    """Writes cfg to a YAML file."""
    with open(path, 'w') as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
