from src.datasets.domain import Domain, make_black_scholes_domain, make_heston_domain
from src.datasets.sampler import BalancedBinSampler, FocusedSampler, UniformSampler

__all__ = [
    "BalancedBinSampler",
    "DatasetGenerator",
    "Domain",
    "FocusedSampler",
    "GeneratedBatch",
    "OptionDataset",
    "UniformSampler",
    "make_black_scholes_domain",
    "make_heston_domain",
]


def __getattr__(name: str):
    if name in {"DatasetGenerator", "GeneratedBatch", "OptionDataset"}:
        from src.datasets.generator import DatasetGenerator, GeneratedBatch, OptionDataset

        values = {
            "DatasetGenerator": DatasetGenerator,
            "GeneratedBatch": GeneratedBatch,
            "OptionDataset": OptionDataset,
        }
        return values[name]
    raise AttributeError(f"module 'src.datasets' has no attribute {name}")
