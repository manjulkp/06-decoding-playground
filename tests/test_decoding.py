import math

import pytest
import torch

from decoding import (
    beam_search,
    filter_logits,
    generate_text,
    repeated_trigram_rate,
    sample_tokens,
)
from experiment import answer_is_correct


class TableModel:
    """A fake model whose next-token probabilities depend only on the last token."""

    def __init__(self, table):
        self.table = torch.tensor(table, dtype=torch.float).clamp_min(1e-9).log()

    def __call__(self, ids, cache=None):
        return self.table[ids[:, -1]], None


class CharTokenizer:
    chars = "abcd"

    def encode(self, text):
        return [self.chars.index(c) for c in text]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


def test_low_temperature_sharpens_and_high_flattens():
    logits = torch.tensor([2.0, 1.0, 0.0])
    top = lambda t: filter_logits(logits, temperature=t).softmax(-1).max().item()
    assert top(0.1) > top(1.0) > top(10.0)
    assert top(0.1) == pytest.approx(1.0, abs=1e-3)
    assert top(1000.0) == pytest.approx(1 / 3, abs=1e-2)


def test_invalid_temperature_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        filter_logits(torch.zeros(3), temperature=0)


def test_top_k_keeps_exactly_k():
    out = filter_logits(torch.tensor([1.0, 4.0, 3.0, 2.0]), top_k=2)
    assert torch.isfinite(out).tolist() == [False, True, True, False]


def test_top_k_larger_than_vocab_keeps_all():
    assert torch.isfinite(filter_logits(torch.tensor([1.0, 2.0]), top_k=99)).all()


def test_top_p_keeps_smallest_group_reaching_p():
    logits = torch.tensor([0.5, 0.3, 0.15, 0.05]).log()
    # 0.5 + 0.3 reaches 0.8, so the third token isn't needed.
    assert torch.isfinite(filter_logits(logits, top_p=0.8)).tolist() == [True, True, False, False]
    assert torch.isfinite(filter_logits(logits, top_p=0.81)).tolist() == [True, True, True, False]


def test_top_p_always_keeps_most_likely_token():
    logits = torch.tensor([0.1, 0.9]).log()
    assert torch.isfinite(filter_logits(logits, top_p=0.01)).tolist() == [False, True]


def test_top_p_keeps_the_right_tokens_when_unsorted_in_a_batch():
    logits = torch.tensor([[0.05, 0.5, 0.15, 0.3], [0.3, 0.05, 0.5, 0.15]]).log()
    kept = torch.isfinite(filter_logits(logits, top_p=0.8))
    assert kept.tolist() == [[False, True, False, True], [True, False, True, False]]


def test_greedy_follows_most_likely_path():
    table = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
    output = sample_tokens(TableModel(table), torch.tensor([[0]]), 5, temperature=0)
    assert output.tolist() == [[0, 1, 2, 3, 0, 1]]


def test_sampling_matches_probabilities():
    table = [[0.7, 0.3], [0.7, 0.3]]
    batch = torch.zeros(5000, 1, dtype=torch.long)
    generator = torch.Generator().manual_seed(0)
    output = sample_tokens(TableModel(table), batch, 1, generator=generator)
    assert (output[:, 1] == 0).float().mean().item() == pytest.approx(0.7, abs=0.03)


def test_same_generator_seed_reproduces_samples():
    table = [[0.5, 0.5], [0.5, 0.5]]
    run = lambda: sample_tokens(
        TableModel(table), torch.zeros(1, 1, dtype=torch.long), 20, generator=torch.Generator().manual_seed(3)
    )
    assert torch.equal(run(), run())


def test_beam_search_beats_greedy_when_greedy_is_trapped():
    # From token 0, token 1 looks best (0.6) but then leads to a coin flip.
    # Token 2 (0.4) leads to a certain next token, so its path has more total probability.
    table = [
        [0, 0.6, 0.4, 0],
        [0.5, 0, 0, 0.5],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ]
    start = torch.tensor([[0]])
    assert sample_tokens(TableModel(table), start, 2, temperature=0).tolist() == [[0, 1, 0]]
    output, score = beam_search(TableModel(table), start, 2, beam_width=2)
    assert output.tolist() == [[0, 2, 3]]
    assert score == pytest.approx(math.log(0.4) / 2)


def test_beam_width_one_equals_greedy():
    table = [[0.1, 0.6, 0.3, 0], [0.3, 0.1, 0.6, 0], [0, 0.3, 0.1, 0.6], [0.6, 0, 0.3, 0.1]]
    start = torch.tensor([[0]])
    greedy = sample_tokens(TableModel(table), start, 6, temperature=0)
    beam, _ = beam_search(TableModel(table), start, 6, beam_width=1)
    assert torch.equal(greedy, beam)


def test_beam_search_stops_at_end_token():
    table = [[0, 0.1, 0.9, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]]
    output, _ = beam_search(TableModel(table), torch.tensor([[0]]), 10, beam_width=2, eos_id=3)
    assert output.tolist() == [[0, 2, 3]]


def test_beam_search_rejects_batches():
    with pytest.raises(ValueError, match="one sequence"):
        beam_search(TableModel([[1.0]]), torch.zeros(2, 1, dtype=torch.long), 1)


CYCLE = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]


def test_generate_text_stops_at_stop_token_and_excludes_it():
    text, reason = generate_text(TableModel(CYCLE), CharTokenizer(), "a", 10, temperature=0, stop_ids={3})
    assert (text, reason) == ("bc", "stop_token")


def test_generate_text_stops_at_string_spanning_tokens():
    text, reason = generate_text(TableModel(CYCLE), CharTokenizer(), "a", 10, temperature=0, stop_strings=["cd"])
    assert (text, reason) == ("b", "stop_string")


def test_generate_text_reports_length_limit():
    text, reason = generate_text(TableModel(CYCLE), CharTokenizer(), "a", 3, temperature=0)
    assert (text, reason) == ("bcd", "length")


def test_repeated_trigram_rate():
    assert repeated_trigram_rate([1, 2, 3, 4, 5]) == 0.0
    assert repeated_trigram_rate([1, 1, 1, 1, 1]) == pytest.approx(2 / 3)
    assert repeated_trigram_rate([1, 2]) == 0.0


@pytest.mark.parametrize(
    "completion, answer, expected",
    [
        (" Paris, the city", "paris", True),
        (" a parisian suburb", "paris", False),
        (" 4.", "four|4", True),
        (" Four", "four|4", True),
        (" fourteen", "four|4", False),
    ],
)
def test_answer_matching(completion, answer, expected):
    assert answer_is_correct(completion, answer) is expected
