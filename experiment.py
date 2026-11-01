"""Compare decoding strategies on bedtime stories, and temperature on factual prompts.

Part 1 measures how repetitive and how varied each strategy's stories are.
Part 2 measures how often GPT-2 gives the right answer to simple facts at each
temperature, which shows why factual outputs should use a low temperature.
"""

import argparse
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from decoding import beam_search, hf_step, repeated_trigram_rate, sample_tokens

MODEL = "openai-community/gpt2"

STORY_PROMPTS = [
    "Once upon a time, there was a sleepy little bear who",
    "Every night, when the moon came out, the tiny mouse",
    "In a cottage at the edge of the forest lived a kind old woman who",
    "The little star looked down at the sleeping village and",
    "Before she went to bed, Lily found a tiny door behind her",
]

STRATEGIES = {
    "Greedy": {"temperature": 0},
    "Beam search (5)": {"beam": 5},
    "Temperature 0.7": {"temperature": 0.7},
    "Temperature 1.0": {"temperature": 1.0},
    "Temperature 1.5": {"temperature": 1.5},
    "Top-k 50": {"temperature": 1.0, "top_k": 50},
    "Top-p 0.9": {"temperature": 1.0, "top_p": 0.9},
}

FACTS = [
    ("The capital of France is", "paris"),
    ("The capital of Japan is", "tokyo"),
    ("The capital of Italy is", "rome"),
    ("The capital of Spain is", "madrid"),
    ("The capital of Germany is", "berlin"),
    ("The capital of Russia is", "moscow"),
    ("The Eiffel Tower is located in the city of", "paris"),
    ("Water is made of hydrogen and", "oxygen"),
    ("The first president of the United States was George", "washington"),
    ("Two plus two equals", "four|4"),
    ("The opposite of hot is", "cold"),
    ("The color of the sky on a clear day is", "blue"),
    ("Christmas is celebrated in the month of", "december"),
    ("The day after Monday is", "tuesday"),
    ("The largest planet in our solar system is", "jupiter"),
    ("The play Romeo and Juliet was written by William", "shakespeare"),
    ("The chemical symbol for gold is", "au"),
    ("There are seven days in a", "week"),
    ("The theory of relativity was developed by Albert", "einstein"),
    ("The currency of Japan is the", "yen"),
]
TEMPERATURES = [0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "axes.titlelocation": "left",
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.7,
    }
)


def answer_is_correct(completion, answer):
    return re.search(rf"\b({answer})\b", completion.lower()) is not None


def compare_with_transformers(model, tokenizer, step):
    """Check that this project's greedy decoder agrees with the library's."""
    ids = torch.tensor([tokenizer.encode(STORY_PROMPTS[0])])
    pad = tokenizer.eos_token_id
    expected = model.generate(ids, max_new_tokens=30, do_sample=False, pad_token_id=pad)
    if not torch.equal(sample_tokens(step, ids, 30, temperature=0), expected):
        raise RuntimeError("greedy decoding does not match transformers")


def run_stories(model, tokenizer, step, new_tokens, samples, generator):
    rows, examples = [], {}
    for name, settings in STRATEGIES.items():
        repeat_rates, overlaps = [], []
        examples[name] = []
        for prompt in STORY_PROMPTS:
            ids = torch.tensor([tokenizer.encode(prompt)])
            if "beam" in settings:
                outputs = [beam_search(step, ids, new_tokens, settings["beam"], tokenizer.eos_token_id)[0][0]]
            elif settings.get("temperature") == 0:
                outputs = [sample_tokens(step, ids, new_tokens, temperature=0)[0]]
            else:
                batch = ids.repeat(samples, 1)
                outputs = list(sample_tokens(step, batch, new_tokens, generator=generator, **settings))
            completions = [output[ids.shape[1] :].tolist() for output in outputs]
            repeat_rates += [repeated_trigram_rate(tokens) for tokens in completions]
            if len(completions) > 1:
                texts = [set(tokens) for tokens in completions]
                overlaps += [
                    len(a & b) / len(a | b)
                    for i, a in enumerate(texts)
                    for b in texts[i + 1 :]
                ]
            examples[name].append(prompt + tokenizer.decode(completions[0]))
        rows.append(
            {
                "name": name,
                "repeated_trigrams": statistics.mean(repeat_rates),
                "sample_overlap": statistics.mean(overlaps) if overlaps else None,
            }
        )
        print(f"{name:<18} repeated trigrams {rows[-1]['repeated_trigrams']:.1%}", flush=True)
    return rows, examples


def run_facts(tokenizer, step, samples, new_tokens, generator):
    rows = []
    for temperature in TEMPERATURES:
        correct, total = 0, 0
        for prompt, answer in FACTS:
            ids = torch.tensor([tokenizer.encode(prompt)])
            # Greedy decoding always gives the same answer, so one sample is enough.
            batch = ids.repeat(1 if temperature == 0 else samples, 1)
            output = sample_tokens(step, batch, new_tokens, temperature=temperature, generator=generator)
            for row in output:
                total += 1
                correct += answer_is_correct(tokenizer.decode(row[ids.shape[1] :]), answer)
        rows.append({"temperature": temperature, "accuracy": correct / total, "n": total})
        print(f"temperature {temperature:<4} accuracy {correct / total:.1%}", flush=True)
    return rows


def plot_facts(rows, path):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = [row["temperature"] for row in rows]
    y = [row["accuracy"] for row in rows]
    ax.plot(x, y, color="#2a78d6", lw=1.6, marker="o", ms=5)
    for index in (0, len(rows) - 1):
        ax.annotate(f"{y[index]:.0%}", (x[index], y[index]), xytext=(0, 8), textcoords="offset points", ha="center", color="#52514e")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set(xlabel="Temperature", ylabel="Facts answered correctly")
    ax.set_title(f"GPT-2 accuracy on {len(FACTS)} simple facts, by temperature")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--story-tokens", type=int, default=60)
    parser.add_argument("--story-samples", type=int, default=3)
    parser.add_argument("--fact-samples", type=int, default=20)
    parser.add_argument("--fact-tokens", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).eval()
    step = hf_step(model)
    compare_with_transformers(model, tokenizer, step)
    print("Greedy decoding matches transformers", flush=True)

    generator = torch.Generator().manual_seed(args.seed)
    story_rows, examples = run_stories(model, tokenizer, step, args.story_tokens, args.story_samples, generator)
    fact_rows = run_facts(tokenizer, step, args.fact_samples, args.fact_tokens, generator)

    lines = [
        "# Decoding results",
        "",
        "## Bedtime stories",
        "",
        f"{len(STORY_PROMPTS)} prompts, {args.story_tokens} new tokens. Sample overlap is how many tokens two "
        "samples from the same prompt share, so a lower number means more variety.",
        "",
        "| Strategy | Repeated three-token sequences | Sample overlap |",
        "| --- | --- | --- |",
    ]
    for row in story_rows:
        overlap = "Always identical" if row["sample_overlap"] is None else f"{row['sample_overlap']:.1%}"
        lines.append(f"| {row['name']} | {row['repeated_trigrams']:.1%} | {overlap} |")
    lines += [
        "",
        "## Simple facts",
        "",
        f"{len(FACTS)} prompts, checking the first {args.fact_tokens} generated tokens for the right answer.",
        "",
        "| Temperature | Correct | Samples |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {row['temperature']} | {row['accuracy']:.1%} | {row['n']} |" for row in fact_rows]
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps({"stories": story_rows, "facts": fact_rows}, indent=1))

    story_lines = ["# Example stories", ""]
    for name, texts in examples.items():
        story_lines += [f"## {name}", ""]
        story_lines += [f"- {text.strip()}".replace("\n", " ") for text in texts]
        story_lines.append("")
    (out / "stories.md").write_text("\n".join(story_lines), encoding="utf-8")
    plot_facts(fact_rows, out / "temperature_vs_facts.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
