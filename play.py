"""Try any decoding settings on any prompt with GPT-2.

Examples:
    uv run python play.py "Once upon a time" --temperature 0
    uv run python play.py "Once upon a time" --temperature 0.8 --top-p 0.9 --stop "."
    uv run python play.py "Once upon a time" --beam 5
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from decoding import beam_search, generate_text, hf_step
from experiment import MODEL


def main(argv=None):
    parser = argparse.ArgumentParser(description="Try decoding settings with GPT-2")
    parser.add_argument("prompt")
    parser.add_argument("--tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=1.0, help="0 means greedy")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--beam", type=int, help="use beam search with this width")
    parser.add_argument("--stop", action="append", default=[], help="stop at this text, can be repeated")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    step = hf_step(AutoModelForCausalLM.from_pretrained(MODEL).eval())
    if args.beam:
        ids = torch.tensor([tokenizer.encode(args.prompt)])
        output, _ = beam_search(step, ids, args.tokens, args.beam, tokenizer.eos_token_id)
        print(tokenizer.decode(output[0]))
        return

    text, reason = generate_text(
        step, tokenizer, args.prompt, args.tokens, args.temperature, args.top_k, args.top_p,
        stop_ids={tokenizer.eos_token_id}, stop_strings=args.stop,
        generator=torch.Generator().manual_seed(args.seed),
    )  # fmt: skip
    print(args.prompt + text)
    print(f"\n[stopped because of: {reason}]")


if __name__ == "__main__":
    main()
