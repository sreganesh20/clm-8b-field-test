# CLM-8B field test

I spent 6 hours trying to build something with [CLM-8B](https://github.com/Contrastive-LM/CLM), the new "System One" model from Contrastive-LM. Instead of generating text, it scores a list of options and returns probabilities. I tried three small projects with it, zero-shot. None of them worked, so this repo is the write-up of why.

![Command-finder results: BM25 vs CLM-8B vs raw embeddings](assets/results.png)

## What I tried

| attempt | what CLM had to decide | baseline | result |
|---|---|---|---|
| **Terminal Bouncer** | Should an AI agent run this shell command: allow, ask first, or block? (16 cases) | 10-line regex | Regex: 14/16 handled acceptably. CLM's best of five framings: 10/16, with 1 dangerous command let through |
| **Tone Radar** | How urgent / passive-aggressive does this message sound? (7 pairs) | raw embeddings | Urgency 3/3, but raw embeddings also 3/3. Passive-aggression 2/4 |
| **Command finder** | Which of 29,852 shell commands does this plain-English request mean? (20 queries) | BM25 keyword search | BM25: 8/20 top-1. CLM: 0/20. As a re-ranker on BM25's top 50, it dropped BM25 from 8/20 to 2/20 |

The speed claims do hold up: about 37 ms for a new decision, and about a millisecond or less when everything is cached.

## The twist

On the README's own quickstart example, my outputs don't match the README, but they *do* match what another user reported in [issue #15](https://github.com/Contrastive-LM/CLM/issues/15). A second open issue, [#3](https://github.com/Contrastive-LM/CLM/issues/3), reports problems with the `Score` question type. So some of what I saw may be about this particular release, or the input format it expects, rather than the idea itself. I can't tell which from my data.

**Scope:** this is about `CLM-v0.1-8B` as released, zero-shot, in these three pipelines. The authors' strongest results use fine-tuned heads, which I didn't try.

**→ The full story, with every number, caveat and dead end, is in [FINDINGS.md](FINDINGS.md).**

## What's in here

```
FINDINGS.md                    the full failure report
assets/results.png             the chart above
make_chart.py                  regenerates the chart
notebooks/
  clm_smoke_test.ipynb         setup + Terminal Bouncer round 1 + Tone Radar + latency
  clm_bouncer_rescue.ipynb     five other ways of asking the Bouncer question
  clm_cmdfinder_gate.ipynb     the command finder + follow-up diagnostics
  catalog.py                   tldr-pages parser, the 20 test queries and their answer regexes
results/                       raw JSON results from the Colab runs
```

## Running it

You need a GPU with roughly 24 GB+ of memory. I used a Colab A100.

1. Open a notebook in Colab and pick an A100 runtime.
2. Run all. The first cells build a separate Python 3.12 environment, because vLLM breaks on Colab's default Python 3.13. They then start the Qwen3-8B encoder with vLLM and start `clm-serve`.
3. Setup takes 5–20 minutes, mostly the model download. Each experiment then runs in a few minutes.

Versions are pinned to what I ran: `contrastive-lm==0.1.0`, `vllm==0.30.0` (torch 2.13.0), Python 3.12, and tldr-pages at commit `106eb6eb`. The Hugging Face model revisions weren't recorded.

## Credits

- CLM code and weights: [Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM) (Apache-2.0).
- Command catalog: [tldr-pages](https://github.com/tldr-pages/tldr) (CC BY 4.0), commit `106eb6eb`. The notebook downloads it; it isn't redistributed here.
- Qwen3-8B: Apache-2.0.
