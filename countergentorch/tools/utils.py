from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, TypeVar, Tuple

import torch
from tqdm import tqdm  # type: ignore
from transformers import GPT2Tokenizer

T = TypeVar("T")


def other(t: Tuple[T, T], x: T) -> T:
    if x == t[0]:
        if x == t[1]:
            raise ValueError(f"{t} contains two copies of {x}")
        return t[1]
    if x != t[1]:
        raise ValueError(f"{t} does not contain {x}")
    return t[0]


def unwrap_or(maybe: Optional[T], default: T) -> T:
    return default if maybe is None else maybe


def all_same(l: Sequence[Any]) -> bool:
    return all(x == l[0] for x in l[1:])


def concat_dicts(dicts: Sequence[Mapping[Any, torch.Tensor]]) -> Dict[Any, torch.Tensor]:
    if not dicts:
        raise ValueError("dicts is empty")
    keys = dicts[0].keys()
    for d in dicts:
        if d.keys() != keys:
            raise ValueError("dicts must have the same keys")
    return {k: torch.cat([d[k] for d in dicts], dim=-1) for k in keys}


def remove_last_tok(d: Mapping[Any, torch.Tensor]) -> Dict[Any, torch.Tensor]:
    return {k: t[:, :-1] for k, t in d.items()}


def maybe_tqdm(it: Iterable[T], do_tqdm: bool = False, **kwargs) -> Iterable[T]:
    if do_tqdm:
        return tqdm(it, **kwargs)
    else:
        return it


def get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_gpt_tokenizer() -> GPT2Tokenizer:
    return GPT2Tokenizer.from_pretrained("gpt2")


def estimate_paraphrase_length(text: str):
    average_token_length = 3
    safety_margin = 50
    return len(text) // average_token_length + safety_margin
