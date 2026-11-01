# Decoding results

## Bedtime stories

5 prompts, 60 new tokens. Sample overlap is how many tokens two samples from the same prompt share, so a lower number means more variety.

| Strategy | Repeated three-token sequences | Sample overlap |
| --- | --- | --- |
| Greedy | 46.2% | Always identical |
| Beam search (5) | 61.7% | Always identical |
| Temperature 0.7 | 4.1% | 14.9% |
| Temperature 1.0 | 0.2% | 9.0% |
| Temperature 1.5 | 0.0% | 1.2% |
| Top-k 50 | 1.6% | 12.1% |
| Top-p 0.9 | 0.7% | 12.3% |

## Simple facts

20 prompts, checking the first 6 generated tokens for the right answer.

| Temperature | Correct | Samples |
| --- | --- | --- |
| 0 | 45.0% | 20 |
| 0.3 | 43.8% | 400 |
| 0.5 | 38.8% | 400 |
| 0.7 | 31.0% | 400 |
| 1.0 | 21.5% | 400 |
| 1.5 | 6.5% | 400 |
| 2.0 | 2.2% | 400 |
