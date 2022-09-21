from math import exp, log2
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, TypeVar, Tuple
import torch


def mean(l: Sequence[float]) -> float:
    return sum(l) / len(l)


def geometric_mean(l: Sequence[float]) -> float:
    return 2 ** (mean(list(map(log2, l))))


def perplexity(log_probs: Sequence[float]):
    """Take in natural log probs, returns (average) perplexity"""
    return exp(mean(log_probs))


def orthonormalize(dir: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Return dir, but projected in the orthogonal of the subspace spanned by dirs.

    Assume that dirs are already orthonomal"""
    inner_products = torch.einsum("n h, h -> n", dirs, dir)
    new_dir = dir - torch.einsum("n, n h -> h", inner_products, dirs)
    new_dir /= torch.linalg.norm(new_dir)

    torch.testing.assert_allclose(torch.einsum("n h, h -> n", dirs, new_dir), torch.zeros(len(dirs)))

    return new_dir
