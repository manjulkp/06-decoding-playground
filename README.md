# Project 6: Decoding playground

A model only gives scores for the next token, and decoding turns those scores into
text. This project writes greedy decoding, beam search, temperature, top-k, top-p and
stop tokens from scratch, then runs them on GPT-2 with bedtime story prompts.

It measures two things:

- **Stories:** how repetitive and how varied each strategy's text is.
- **Simple facts:** how often GPT-2 answers correctly at each temperature.

## Results

Lower temperatures answered simple facts correctly more often, falling from 45.0% at
temperature 0 to 2.2% at temperature 2.0. Greedy and beam search were the most
repetitive story strategies, and sampling at higher temperatures reduced repetition.

| Strategy | Repeated three-token sequences | Sample overlap |
| --- | --- | --- |
| Greedy | 46.2% | Always identical |
| Beam search (5) | 61.7% | Always identical |
| Temperature 0.7 | 4.1% | 14.9% |
| Temperature 1.0 | 0.2% | 9.0% |
| Temperature 1.5 | 0.0% | 1.2% |
| Top-k 50 | 1.6% | 12.1% |
| Top-p 0.9 | 0.7% | 12.3% |

| Temperature | Correct | Samples |
| --- | --- | --- |
| 0 | 45.0% | 20 |
| 0.3 | 43.8% | 400 |
| 0.5 | 38.8% | 400 |
| 0.7 | 31.0% | 400 |
| 1.0 | 21.5% | 400 |
| 1.5 | 6.5% | 400 |
| 2.0 | 2.2% | 400 |

Full results are in `outputs\results.md`, `outputs\stories.md` and
`outputs\temperature_vs_facts.png`. One Temperature 0.7 sample was removed from
`outputs\stories.md` because GPT-2 generated inappropriate text.

## Link to your own LLM features

This project is meant to help you explain why you'd use a low temperature for factual
outputs like `explain_alert.py`. Use the temperature results as evidence.
GPT-2 is a small, old model, so they show the general effect and don't predict how
any specific modern model will behave.

## Run it

```powershell
uv sync
uv run python experiment.py       # downloads GPT-2 the first time, about 500 MB
uv run pytest
```

Try your own settings:

```powershell
uv run python play.py "Once upon a time" --temperature 0
uv run python play.py "Once upon a time" --temperature 0.8 --top-p 0.9 --stop "."
uv run python play.py "Once upon a time" --beam 5
```

## Files

| File | What it does |
| --- | --- |
| `decoding.py` | All the decoding strategies, which work with any model |
| `experiment.py` | Runs the story and fact comparisons |
| `play.py` | Tries any settings on any prompt |
| `tests/` | Tests each strategy using a small fake model |

Each run of `experiment.py` also checks that this project's greedy decoding matches
the Hugging Face `transformers` library, and stops if it doesn't.
