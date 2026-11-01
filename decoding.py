"""Decoding strategies, written from scratch.

A model only gives scores (logits) for the next token. Decoding is how those scores
become text. Everything here talks to the model through a `step` function:

    logits, cache = step(ids, cache)

`ids` holds the new tokens, `logits` has shape (batch, vocab) for the next token, and
`cache` is whatever the model wants passed back on the next call.
"""

import torch
from torch.nn import functional as F


def hf_step(model):
    """Create a step function for a Hugging Face model, with its KV cache."""

    @torch.no_grad()
    def step(ids, cache=None):
        output = model(input_ids=ids, past_key_values=cache, use_cache=True)
        return output.logits[:, -1, :], output.past_key_values

    return step


def filter_logits(logits, temperature=1.0, top_k=None, top_p=None):
    """Apply temperature, then remove unlikely tokens by setting them to -infinity."""
    if temperature <= 0:
        raise ValueError("temperature must be positive; use temperature=0 only with sample_tokens")
    # Dividing by a small temperature widens the gaps between scores, which sharpens
    # the distribution. A large temperature flattens it.
    logits = logits / temperature
    if top_k is not None:
        threshold = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p is not None:
        # Keep the smallest group of likely tokens whose total probability reaches
        # top_p. Unlike top-k, the group grows when the model is unsure.
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Compare the total *before* each token, so that the first token is always kept.
        remove_sorted = (cumulative - F.softmax(sorted_logits, dim=-1)) >= top_p
        remove = remove_sorted.scatter(-1, order, remove_sorted)
        logits = logits.masked_fill(remove, float("-inf"))
    return logits


def sample_tokens(step, ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None, generator=None):
    """Generate a fixed number of tokens per row. Temperature 0 means greedy decoding."""
    cache, current = None, ids
    for _ in range(max_new_tokens):
        logits, cache = step(current, cache)
        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            filtered = filter_logits(logits, temperature, top_k, top_p)
            next_id = torch.multinomial(F.softmax(filtered, dim=-1), 1, generator=generator)
        ids = torch.cat([ids, next_id], dim=1)
        current = next_id
    return ids


def generate_text(
    step, tokenizer, prompt, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
    stop_ids=(), stop_strings=(), generator=None,
):  # fmt: skip
    """Generate one completion, ending early at a stop token or stop string.

    Returns (text, reason), where reason is "stop_token", "stop_string" or "length".
    The stop token or string itself isn't included in the text.
    """
    current = torch.tensor([tokenizer.encode(prompt)])
    cache = None
    new_ids = []
    for _ in range(max_new_tokens):
        logits, cache = step(current, cache)
        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            filtered = filter_logits(logits, temperature, top_k, top_p)
            next_id = torch.multinomial(F.softmax(filtered, dim=-1), 1, generator=generator)
        token = next_id.item()
        if token in stop_ids:
            return tokenizer.decode(new_ids), "stop_token"
        new_ids.append(token)
        # Decode and check the text, because a stop string can span several tokens.
        text = tokenizer.decode(new_ids)
        for stop in stop_strings:
            if stop in text:
                return text[: text.index(stop)], "stop_string"
        current = next_id
    return tokenizer.decode(new_ids), "length"


def beam_search(step, ids, max_new_tokens, beam_width=5, eos_id=None, length_alpha=1.0):
    """Keep the `beam_width` best partial sequences at each step and return the best.

    Scores are total log-probabilities, which get more negative as text gets longer,
    so finished sequences are divided by length**length_alpha to compare them fairly.
    Input must be a single sequence. Returns (ids, score).
    """
    if ids.shape[0] != 1:
        raise ValueError("beam_search takes one sequence at a time")
    prompt_length = ids.shape[1]
    beams = [(ids[0], 0.0)]
    finished = []

    def normalized(sequence, score):
        return score / max(1, len(sequence) - prompt_length) ** length_alpha

    for _ in range(max_new_tokens):
        batch = torch.stack([sequence for sequence, _ in beams])
        logits, _ = step(batch, None)  # beams change order, so the cache isn't reused
        log_probs = F.log_softmax(logits, dim=-1)

        candidates = []
        for (sequence, score), row in zip(beams, log_probs):
            values, indices = row.topk(min(beam_width, row.shape[-1]))
            for value, index in zip(values.tolist(), indices.tolist()):
                candidates.append((sequence, index, score + value))
        candidates.sort(key=lambda item: item[2], reverse=True)

        beams = []
        for sequence, token, score in candidates[: beam_width * 2]:
            extended = torch.cat([sequence, torch.tensor([token])])
            if token == eos_id:
                finished.append((extended, score))
            elif len(beams) < beam_width:
                beams.append((extended, score))
        if not beams:
            break

    finished.extend(beams)
    best = max(finished, key=lambda item: normalized(*item))
    return best[0].unsqueeze(0), normalized(*best)


def repeated_trigram_rate(tokens):
    """The share of three-token sequences that already appeared earlier in the text."""
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    if not trigrams:
        return 0.0
    return 1 - len(set(trigrams)) / len(trigrams)
