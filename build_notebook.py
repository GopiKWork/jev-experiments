"""Generate model_architecture.ipynb.

Keeping the notebook's cells in a plain Python file makes them reviewable in a
diff. Run this script to regenerate the notebook, then execute the notebook to
fill in the outputs.
"""

import json
from pathlib import Path

cells = []


def md(text):
    cells.append(
        {"cell_type": "markdown", "id": f"md{len(cells)}", "metadata": {}, "source": text.strip().splitlines(keepends=True)}
    )


def code(text):
    cells.append(
        {
            "cell_type": "code",
            "id": f"code{len(cells)}",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.strip().splitlines(keepends=True),
        }
    )


md("""
# How a System 1 Decision Model Differs From an LLM

A System 1 decision model answers a typed question in one forward pass and returns a
probability distribution over the options you supplied. It never generates text. This
notebook opens two of them, Laya and Decider, alongside a conventional LLM, and locates
the exact place where the three architectures diverge.

The three models are deliberately different in size and in family, because the point is
not that a small model is cheaper. The point is that all three answer the same question
type, and only one of them does it by sampling tokens.

| Model | Family | Parameters | Direction | Answer comes from |
|---|---|---|---|---|
| Laya | ModernBERT-large encoder | 421M | Bidirectional | A purpose-built scorer head |
| Decider-4b | Qwen3.5 hybrid decoder | 4.2B | Causal | The LM head, sliced to option labels |
| gpt-oss-20b | Mixture-of-experts decoder | 21B | Causal | The full vocabulary head, sampled in a loop |

The notebook is ordered so the cheap work comes first. Sections 1 to 4 read configuration
files and safetensors headers, which needs no GPU and almost no memory. Sections 5 and 6
load the weights and run instrumented forward passes.

**What you need.** The `openrouter` virtual environment from this directory's README, with
`laya` and `decider-ai` installed. About 9 GB of model weights, downloaded on first use.
No GPU. The gpt-oss column costs a few kilobytes, because we read its `config.json` and
never download its weights.
""")

md("""
## 1. Setup

One helper does most of the work in the early sections. `safetensors` files begin with a
JSON header that lists every tensor, its shape, and its dtype. Reading that header tells
you the full layer inventory of a model without allocating a single parameter.
""")

code('''
import json
import os
import struct
from collections import OrderedDict
from math import prod
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # keep download bars out of the outputs

from huggingface_hub import hf_hub_download, snapshot_download
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()

LAYA = "convaiinnovations/laya"
DECIDER, DECIDER_REV = "Mapika/decider-4b", "v2"
BASELINE = "openai/gpt-oss-20b"  # config only, for the comparison column


def safetensors_header(path):
    """Return the tensor index of a safetensors file without reading any weights."""
    with open(path, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(length))
    return {k: v for k, v in header.items() if k != "__metadata__"}


def params(entry):
    return prod(entry["shape"]) if entry["shape"] else 1


def table(rows, headers):
    """Print a left-aligned text table. Keeps the notebook free of plotting dependencies."""
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


laya_dir = Path(snapshot_download(LAYA))
decider_dir = Path(snapshot_download(DECIDER, revision=DECIDER_REV))
print("laya   ", laya_dir)
print("decider", decider_dir)
''')

md("""
## 2. Three architectures at a glance

Every Hugging Face model carries a `config.json`. Read three of them side by side and the
family differences appear before any weight is touched.
""")

code('''
laya_encoder_cfg = json.loads((laya_dir / "encoder" / "config.json").read_text())
laya_agent_cfg = json.loads((laya_dir / "rl_agent_config.json").read_text())
decider_cfg = json.loads((decider_dir / "config.json").read_text())
baseline_cfg = json.loads(Path(hf_hub_download(BASELINE, "config.json")).read_text())


def layer_mix(cfg):
    """Summarise the layer_types list as counts, or report uniform attention."""
    types = cfg.get("layer_types")
    if not types:
        return "full_attention x %d" % cfg["num_hidden_layers"]
    counts = OrderedDict()
    for t in types:
        counts[t] = counts.get(t, 0) + 1
    return ", ".join(f"{t} x {n}" for t, n in counts.items())


rows = []
for name, cfg in (("Laya encoder", laya_encoder_cfg), ("Decider-4b", decider_cfg), ("gpt-oss-20b", baseline_cfg)):
    rows.append(
        [
            name,
            cfg["architectures"][0],
            cfg["num_hidden_layers"],
            cfg["hidden_size"],
            f"{cfg['vocab_size']:,}",
            f"{cfg['max_position_embeddings']:,}",
            layer_mix(cfg),
        ]
    )
table(rows, ["model", "architecture", "layers", "hidden", "vocab", "max pos", "layer mix"])
''')

md("""
Three things in that table matter for the rest of the notebook.

**Laya is an encoder.** `ModernBertForMaskedLM` is a masked language model, so every token
attends to every other token in both directions. There is no causal mask and no notion of
"the next token". That is a natural fit for a model whose job is to read a fixed document
and score a fixed set of options.

**Decider is a hybrid, not a plain decoder.** Its `layer_types` list is 24 `linear_attention`
layers interleaved with 8 `full_attention` layers, one full layer after every three linear
ones. A linear-attention layer carries a recurrent state instead of a quadratic attention
matrix, so there is no per-layer attention map to inspect for three quarters of the stack.
This is the Qwen3.5 gated-delta-rule design, and it is the reason the model needs either a
Triton kernel or the slower reference PyTorch path.

**gpt-oss is a mixture of experts.** `num_local_experts` is 32 with `experts_per_token` 4,
so only an eighth of its feed-forward capacity runs per token. Its vocabulary is 201,088
entries, which is where the next section's argument begins.
""")

code('''
print("gpt-oss experts:          ", baseline_cfg["num_local_experts"], "total,", baseline_cfg["experts_per_token"], "active per token")
print("Laya sequence budget:     ", laya_agent_cfg["max_len"], "tokens (config allows", laya_encoder_cfg["max_position_embeddings"], ")")
print("Laya decision head layers:", laya_agent_cfg["head_layers"])
print("Decider option ceiling:   ", json.loads((decider_dir / "decider_config.json").read_text())["max_options"], "options per question")
''')

md("""
## 3. Weight inventory, with no model loaded

Now read the two safetensors headers. Grouping tensor names by prefix shows how each model
spends its parameters, and in particular how much of each model is the language backbone
and how much is the decision machinery.
""")

code('''
def inventory(path, depth=2):
    """Group tensors by the first `depth` dot-separated name components."""
    header = safetensors_header(path)
    groups = OrderedDict()
    for name, entry in sorted(header.items()):
        key = ".".join(name.split(".")[:depth])
        n, p = groups.get(key, (0, 0))
        groups[key] = (n + 1, p + params(entry))
    total = sum(p for _, p in groups.values())
    return groups, total, len(header)


laya_groups, laya_total, laya_n = inventory(laya_dir / "model.safetensors")
table(
    [[k, n, f"{p:,}", f"{100 * p / laya_total:.1f}%"] for k, (n, p) in laya_groups.items()],
    ["group", "tensors", "params", "share"],
)
print(f"\\nLaya total: {laya_total:,} parameters in {laya_n} tensors")
''')

code('''
decider_groups, decider_total, decider_n = inventory(decider_dir / "model.safetensors", depth=3)
table(
    [[k, n, f"{p:,}", f"{100 * p / decider_total:.1f}%"] for k, (n, p) in decider_groups.items()],
    ["group", "tensors", "params", "share"],
)
print(f"\\nDecider total: {decider_total:,} parameters in {decider_n} tensors")
''')

md("""
This is the first real result, and the two models answer the same question in opposite ways.

Laya's inventory has named decision components next to the encoder: a `head` of transformer
layers, a `scorer`, a `type_emb`, an `act_head`, and a `temperature` buffer. Add them up and
the decision machinery is a small fraction of the model; the rest is the ModernBERT encoder
it was fine-tuned from.

Decider's inventory has nothing but `model.language_model`. There is no decision head in the
checkpoint at all. Decider adds **zero** parameters to its Qwen3.5 backbone. Its typed
answers come entirely from how the existing weights are read at inference time, which is
what the next section shows.
""")

code('''
decision = {k: p for k, (n, p) in laya_groups.items() if not k.startswith("encoder")}
print("Laya decision machinery:")
for k, p in decision.items():
    print(f"  {k:24s} {p:>12,}")
print(f"  {'total':24s} {sum(decision.values()):>12,}  ({100 * sum(decision.values()) / laya_total:.1f}% of the model)")

backbone = laya_total - sum(decision.values())
print(f"\\nLaya encoder backbone:     {backbone:>12,}")
print(f"Decider decision machinery: {0:>12,}  (0.0% of the model)")
''')

md("""
## 4. Where the answer distribution comes from

This is the section that explains the model class. All three models produce a probability
distribution from a hidden state, and they differ only in what that distribution is over.

Start with Laya. Its input is not a chat transcript. It is a single sequence that packs the
question, the options, and the state together, with a `[MASK]` token placed in front of
every option. Those mask positions are the option markers.
""")

code('''
import inspect

import laya.common as lc

print(inspect.getdoc(lc.build_sequence).splitlines()[0])
''')

md("""
The model runs the encoder over that sequence, gathers the hidden state at each option
marker, and pushes each one through a scorer that returns a single number. Softmax over
those numbers is the answer. Here is the head, verbatim from the installed package.
""")

code('''
src = inspect.getsource(lc.DecisionModel.forward)
core = [ln for ln in src.splitlines() if any(k in ln for k in ("last_hidden_state", "gather", "self.scorer", "masked_fill", "type_emb"))]
print("\\n".join(core))
print()
print("scorer:  ", lc.DecisionModel(type("C", (), {"config": type("c", (), {"hidden_size": 1024})})()).scorer)
''')

md("""
Read that carefully. `torch.gather` picks out the marker positions, `self.scorer` maps each
1024-dimensional hidden state to one scalar, and `masked_fill` removes the slots that this
particular question does not use. The distribution is over *your options*, and its width
changes from question to question. Nothing about the vocabulary is involved.

Decider reaches the same shape of output by a completely different route. It keeps the
Qwen3.5 language-model head and throws away almost all of it.
""")

code('''
from decider.model import DecisionModel as DeciderModel

print(inspect.getsource(DeciderModel.slot_logits))
''')

md("""
Line by line: `self.lm.model(...)` runs the backbone and stops, so the language-model head
is never applied to the whole sequence. `h[slot_batch, slot_idx]` gathers the hidden state at
one answer slot per question, exactly as Laya gathers at its markers.
`self.lm.lm_head.weight[self.letters]` is the trick: it selects only the rows of the output
embedding matrix that correspond to the option label tokens, `A`, `B`, `C`, and so on.
`masked_fill` then removes labels beyond the option count, and a softmax over what is left is
the answer.

So Decider's answer is a distribution over single-letter tokens, renormalized to the options
you actually supplied. Let us see how big that slice is.
""")

code('''
from transformers import AutoTokenizer

from decider.prompt import MAX_OPTIONS, letter_ids

tok = AutoTokenizer.from_pretrained(decider_dir)
letters = letter_ids(tok)
vocab = decider_cfg["vocab_size"]
hidden = decider_cfg["hidden_size"]

print(f"full LM head:    {vocab:>7,} x {hidden:,}  = {vocab * hidden:>13,} weights")
print(f"sliced to labels:{len(letters):>7,} x {hidden:,}  = {len(letters) * hidden:>13,} weights")
print(f"fraction used:   {100 * len(letters) / vocab:.3f}% of the vocabulary")
print(f"\\nfirst 12 label tokens: {[tok.decode([i]) for i in letters[:12]]}")
print(f"option ceiling:        {MAX_OPTIONS}")
''')

md("""
A conventional LLM uses the entire matrix on the left, then samples a token from it, appends
that token, and runs the whole model again. That loop is the third architecture in the
comparison, and it is the only one of the three whose cost depends on how long the answer is.

| | Laya | Decider | gpt-oss |
|---|---|---|---|
| Output space | Your options | Option label tokens, renormalized | Full vocabulary |
| Decision parameters | Trained head and scorer, 26.5M | Reused, zero added | None, the head is the vocabulary |
| Forward passes per answer | 1 | 1 | One per generated token |
| Answer needs parsing | No | No | Yes |
| Probability calibrated | Yes, per question type | Yes, per question type | No |

The last row is not a detail. Both decision models ship a fitted temperature and apply it
before the softmax, which is what makes their confidences comparable across questions.
Section 6 measures that.
""")

md("""
## 5. Attention: three shapes of mask

The direction of attention is the other structural difference. Laya has no causal mask at
all, so a token near the start of the state can attend to the customer's final sentence.
Decider is causal, so information only flows forwards. Build both masks and look at them.
""")

code('''
import torch

T = 8


def show(mask, title):
    print(title)
    for row in mask:
        print("   " + " ".join("#" if v else "." for v in row.tolist()))
    print(f"   visible pairs: {int(mask.sum())} of {mask.numel()}\\n")


show(torch.ones(T, T, dtype=torch.bool), "Laya, bidirectional (ModernBERT full_attention layer)")
show(torch.tril(torch.ones(T, T, dtype=torch.bool)), "Decider, causal (Qwen3.5 full_attention layer)")

band = (torch.arange(T)[:, None] - torch.arange(T)[None, :]).abs() <= 2
show(band, "Laya, sliding window (ModernBERT sliding_attention layer, window narrowed to 2 to fit)")
''')

md("""
The real sliding window is wider than the illustration. ModernBERT applies `local_attention`
tokens of context on the layers between its global ones, and promotes one layer to full
attention every `global_attn_every_n_layers`.

Decider's linear-attention layers have no mask to draw, which is the point. Instead of an
attention matrix they carry a recurrent state that is updated token by token, so their cost
is linear in sequence length rather than quadratic. Count the two kinds in each model.
""")

code('''
for name, cfg in (("Laya encoder", laya_encoder_cfg), ("Decider-4b", decider_cfg), ("gpt-oss-20b", baseline_cfg)):
    types = cfg.get("layer_types") or ["full_attention"] * cfg["num_hidden_layers"]
    quadratic = sum(1 for t in types if t == "full_attention")
    print(f"{name:14s} {len(types):>3} layers, {quadratic:>3} with a quadratic attention matrix "
          f"({100 * quadratic / len(types):.0f}%)")
print()
print("Laya  sliding window", laya_encoder_cfg["local_attention"], "tokens, global attention every",
      laya_encoder_cfg["global_attn_every_n_layers"], "layers")
print("Decider full attention every", decider_cfg["full_attention_interval"], "layers")
''')

md("""
## 6. One instrumented forward pass

Everything so far came from headers and source. Now load the models and watch the tensors
move. Laya first, because it is small.

The hooks below print the shape of the encoder output and of the scorer's input and output.
Those three shapes are the whole architecture in miniature.
""")

code('''
import warnings

os.environ.setdefault("USE_TF", "0")
warnings.filterwarnings("ignore", module="laya")

import laya

STATE = (
    "My 2024 sedan's brakes started grinding this morning and the pedal "
    "feels soft. It's still under warranty. Can I bring it in today?"
)
QUESTION = {
    "department": {
        "type": "choice",
        "instructions": "Which dealership department should handle this?",
        "criteria": {"Service": None, "Parts": None, "Sales": None, "Warranty Claims": None},
    }
}

agent = laya.load(LAYA)
seen = {}
hooks = [
    agent.model.encoder.register_forward_hook(lambda m, i, o: seen.update(encoder=tuple(o.last_hidden_state.shape))),
    agent.model.scorer.register_forward_hook(lambda m, i, o: seen.update(scorer_in=tuple(i[0].shape), scorer_out=tuple(o.shape))),
]
answer = agent.predict(STATE, QUESTION)["answers"]["department"]
for h in hooks:
    h.remove()

print("encoder output        [batch, tokens, hidden] ", seen["encoder"])
print("scorer input          [batch, options, hidden]", seen["scorer_in"])
print("scorer output         [batch, options, 1]     ", seen["scorer_out"])
print("\\nanswer:", answer["choice"], "with probabilities", answer["probabilities"])
''')

md("""
The scorer received one hidden state per option and returned one number per option. The
4-way distribution at the end is a softmax over exactly those four numbers. If you added a
fifth department, the scorer output would be `[1, 5, 1]` and nothing else in the model would
change.

Now Decider. Rather than calling the public API, reproduce its head by hand so that every
intermediate is visible.
""")

code('''
import sys

sys.modules.setdefault("fla", None)  # its Triton kernel needs a GPU, so take the torch path

del agent  # free the Laya weights before loading a 4.2B model

from decider.infer import Decider
from decider.model import collate

d = Decider(decider_dir)

rqs, index, items = d._system_one_items(STATE, QUESTION)
item = items[0]
print("rendered prompt, tail 300 characters:")
print("   ..." + d.m.tok.decode(item["ids"])[-300:].replace("\\n", "\\n   "))
print("\\nanswer slot position:", item["slots"], "of", len(item["ids"]), "tokens")
print("options at that slot:", item["nopts"])
''')

code('''
batch = collate(items, d.m.tok.pad_token_id)
with torch.no_grad():
    h = d.m.lm.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
print("backbone output [batch, tokens, hidden]", tuple(h.shape))

slot = h[batch["slot_batch"], batch["slot_idx"]]
print("hidden state at the answer slot       ", tuple(slot.shape))

W_full = d.m.lm.lm_head.weight
W_slice = W_full[d.m.letters]
print("full LM head                          ", tuple(W_full.shape))
print("head sliced to option labels          ", tuple(W_slice.shape))

logits = (slot.float() @ W_slice.float().T)
print("logits over labels                    ", tuple(logits.shape))
''')

md("""
The last three lines are the difference between a decision model and a language model,
expressed as tensor shapes. A language model would multiply that same hidden state by the
full matrix and get one logit per vocabulary entry. Decider multiplies by 255 rows and gets
one logit per possible option label.

Mask the labels this question does not use, divide by the fitted temperature, and softmax.
""")

code('''
valid = logits[0, : item["nopts"][0]]
names = rqs["department"]["names"]

raw = torch.softmax(valid, -1)
calibrated = torch.softmax(valid / d.T, -1)

table(
    [[n, f"{v:+.2f}", f"{r:.3f}", f"{c:.3f}"] for n, v, r, c in zip(names, valid.tolist(), raw.tolist(), calibrated.tolist())],
    ["option", "logit", "softmax", f"softmax at T={d.T}"],
)
print("\\npublic API agrees:", d.system_one(STATE, QUESTION)["answers"]["department"]["probabilities"])
''')

md("""
## 7. Calibration temperature, the invisible component

A conventional LLM gives you a token distribution shaped by sampling parameters you choose
at call time. Neither decision model works that way. Both ship a temperature that was
**fitted after training**, on held-out data, and both apply a different one per question
type, because a two-option yes/no question and a five-level rubric are differently
overconfident.

This is a component with no equivalent in a standard LLM architecture, and it lives in the
config rather than the weights.
""")

code('''
print("Decider, from decider_config.json")
dc = json.loads((decider_dir / "decider_config.json").read_text())
print("   temperature       ", dc["temperature"])
print("   fitted by         ", dc["stage"].split("temperature fitted by")[-1].strip()[:90])

print("\\nLaya, from rl_agent_config.json")
for bucket, t in sorted(laya_agent_cfg["temperature_by_options"].items()):
    print(f"   {bucket:12s} {t:.4f}")
print("\\nLaya also stores the temperature in the checkpoint itself:",
      [k for k in safetensors_header(laya_dir / 'model.safetensors') if 'temperature' in k])
''')

code('''
print("Effect of Decider's temperature on the department answer:\\n")
table(
    [
        ["uncalibrated (T=1)", f"{raw.max():.3f}"],
        [f"calibrated (T={d.T})", f"{calibrated.max():.3f}"],
    ],
    ["setting", "top probability"],
)
print("\\nA temperature above 1 flattens the distribution. The model was overconfident,")
print("and the fitted value is what makes the number usable as a threshold in code.")
''')

md("""
## 8. What this means in practice

Three architectures, one task, and the difference is entirely in the last few tensor
operations.

- A **standard LLM** ends in a full-vocabulary matrix and an autoregressive loop. To get a
  label out of it you prompt it, sample tokens, and parse the result. The probability it
  reports is over tokens, not over your options, so it does not threshold cleanly.
- **Laya** replaces the head entirely. It is an encoder, so it reads the whole state in both
  directions, and a purpose-built scorer turns the hidden state at each option marker into
  one number. The decision machinery is a few percent of a 421M-parameter model, which is
  why it runs in half a second on a CPU.
- **Decider** keeps the head and slices it. It adds no parameters at all; the typed answer
  comes from reading an existing Qwen3.5 backbone at one position and restricting the output
  matrix to 255 label rows. That buys it a much stronger backbone at the cost of ten times
  the parameters.

Both decision models ship a fitted temperature per question type, and both return a
distribution whose width is set by your question rather than by the tokenizer. That is what
makes the output safe to compare against a threshold in ordinary code, which is the pattern
the other scripts in this directory use for steering.

The size difference shows up in accuracy. `benchmark.py` routes 50 dealership scenarios
through an agent steered by each model, and on the same 8-core CPU the two open-weight models
separate cleanly: Laya reaches 72 percent at about 0.5 seconds per call, Decider reaches 94
percent at about 3 seconds. The 10x parameter count buys 22 points. Whether that trade is
worth it depends entirely on your latency budget, which is the practical reason to know which
of these two architectures you are holding.

### Where to look next

- `jev_basics.py`, `laya_basics.py`, and `decider_basics.py` in this directory send the same
  three questions to the three models.
- `benchmark.py` measures what the accuracy difference is worth across 50 scenarios.
- The Laya head is `laya/common.py`, `DecisionModel.forward`. Decider's is
  `decider/model.py`, `DecisionModel.slot_logits`. Both are short enough to read in full.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).with_name("model_architecture.ipynb")
out.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"wrote {out} with {len(cells)} cells")
