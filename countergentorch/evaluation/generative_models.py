from math import exp
from typing import TYPE_CHECKING, Callable, Sequence, List, Optional, Tuple

import torch
from countergen.tools.math_utils import perplexity
from countergentorch.tools.utils import concat_dicts, get_device, get_gpt_tokenizer, remove_last_tok
from countergen.types import Input, ModelEvaluator, Outputs, Performance
from transformers import BatchEncoding, GPT2LMHeadModel, GPT2Tokenizer

metrics = ["perplexity", "probability"]

LogProbs = float

# Return a log prob for each token of output
GenerativeModel = Callable[[Input, Outputs], Sequence[Sequence[LogProbs]]]


def pt_to_generative_model(model: torch.nn.Module) -> GenerativeModel:
    """Make a GenerativeModel out of a pytorch model.

    The model should take {"input_ids": [tensor], "attention_mask": [tensor]} as input,
    and return something that has a "logits" attribute."""

    tokenizer = get_gpt_tokenizer()

    def gen_model(inp: Input, out: Outputs) -> List[List[float]]:
        tokens_inp = tokenizer(inp, return_tensors="pt").to(model.device)
        token_outs = [tokenizer(o, return_tensors="pt").to(model.device) for o in out]

        return get_correct_logprobs(tokens_inp, token_outs, model)

    return gen_model


def get_generative_model_evaluator(model: GenerativeModel, metric: str = "probability") -> ModelEvaluator:
    """Return the ModelEvaluator corresponding to the model & the metric.

    Available metrics: probability & perplexity"""

    def run(inp: Input, out: Outputs) -> Performance:
        if len(out) == 0:
            raise ValueError("Expected output should be provided for gpt models")
        if len(inp) == 0:
            raise ValueError("Empty inputs are forbidden for gpt models.")

        correct_log_probs_list = model(inp, out)

        total_prob: float = 0
        total_log_prob: float = 0
        number_of_toks: int = 0
        for correct_log_probs in correct_log_probs_list:
            total_prob += exp(sum(correct_log_probs))
            number_of_toks += len(correct_log_probs)
            total_log_prob += sum(correct_log_probs)

        if metric == "perplexity":
            return exp(-total_log_prob / number_of_toks)
        if metric == "probability":
            return total_prob
        raise ValueError(f"{metric} is not a valid metric. Choose one in {metrics}.")

    return run


def get_correct_logprobs(
    tokens_inp: BatchEncoding, token_outs: List[BatchEncoding], model: torch.nn.Module
) -> List[List[float]]:

    if all([o["input_ids"].shape[-1] == 1 for o in token_outs]):
        return [[x.item() for x in t] for t in get_correct_1tok_logprobs(tokens_inp, token_outs, model)]

    inp_length = tokens_inp["input_ids"].shape[-1]

    result: List[List[float]] = []

    for tokens_out in token_outs:
        out_length = tokens_out["input_ids"].shape[-1]
        assert out_length > 0, "Zero length expected output is forbidden"

        tokens_to_feed = remove_last_tok(concat_dicts([tokens_inp, tokens_out]))
        with torch.no_grad():
            logits = model(**tokens_to_feed).logits[0].to("cpu")
        log_probs = torch.log_softmax(logits, dim=-1)[inp_length - 1 :, :]

        assert len(log_probs) == len(tokens_out["input_ids"][0])
        correct_log_probs = torch.gather(log_probs, 1, tokens_out["input_ids"][0, :, None])[:, 0]

        result.append([x.item() for x in correct_log_probs])

    return result


def get_correct_1tok_logprobs(
    tokens_inp: BatchEncoding, token_outs: List[BatchEncoding], model: torch.nn.Module
) -> List[torch.Tensor]:

    with torch.no_grad():
        logits = model(**tokens_inp).logits[0].to("cpu")
    log_probs = torch.log_softmax(logits, dim=-1)[-1:, :]

    good_tokens = [o["input_ids"][0, 0].item() for o in token_outs]

    correct_log_probs = [log_probs[:, i] for i in good_tokens]

    return correct_log_probs
