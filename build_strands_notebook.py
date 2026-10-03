"""Generate strands_decider_architecture.ipynb.

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
# Inside the Strands Decision Model

Strands published an open-weight System 1 decision model on 30 September 2026:
`StrandsAgents/strands-decider-2B-hobson-v19`. It answers the same three typed
questions as Jev, Laya and Decider, and it never generates text.

It is worth opening because it reads its answer out of a place neither of the models in
`model_architecture.ipynb` does. Laya is an encoder with a classification head. Decider-4b
is a decoder that reads the next-token distribution and keeps only the option-label
tokens. This model is a decoder that ignores the next-token distribution entirely: a
pointer head scores each option from the hidden state sitting at the end of that option's
own line in the prompt.

Two corrections worth making up front, because both are easy assumptions:

- **It is not a BERT model.** It is a LoRA adapter on `Qwen/Qwen3.5-2B-Base`, a causal
  decoder. The confusion is understandable, because an unaffiliated ModernBERT decision
  model (`altslate/certo-decision-model`) appeared ten days earlier.
- **"Decoder" does not imply the next-token trick.** That sentence is true of Decider-4b
  and false here. Sections 4 and 5 show where this model reads instead.

| | Laya | Decider-4b | strands-decider-2B |
|---|---|---|---|
| Torso | ModernBERT-large encoder | Qwen3.5 hybrid decoder | Qwen3.5-2B hybrid decoder |
| Parameters | 421M | 4.2B | 2B, of which 17.9M trained |
| Answer comes from | A classification head | The LM head, sliced to option tokens | A pointer head over option positions |
| Option count ceiling | Fixed by the head | Fixed by the prompt | None |
| Calibration | One temperature per question kind | One scalar, 1.935 | One temperature per question kind |

**What you need.** The virtual environment from this directory's README, with
`strands-decider` installed. About 4.5 GB of weights, downloaded on first use. No GPU.
Sections 1 to 3 read files and need almost no memory. Sections 4 to 6 load the weights.
""")

md("""
## 1. Setup

One import ordering matters. Qwen3.5 is a hybrid: most of its layers use linear attention
rather than softmax attention. When `flash-linear-attention` is importable, transformers
routes those layers through a Triton kernel that needs a GPU driver. Hiding the package
before transformers loads selects the reference PyTorch path, which is correct and slower.
Remove the `sys.modules` line on a GPU.
""")

code('''
import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.modules["fla"] = None  # force the CPU path; delete this line on a GPU

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, safe_open

MODEL = "StrandsAgents/strands-decider-2B-hobson-v19"

checkpoint = snapshot_download(MODEL)
print(checkpoint)
''')

md("""
## 2. What the checkpoint actually contains

The published repo is not a model in the usual sense. There is no `config.json` and no
`model.safetensors` at the root, because the 2B of weights are not here at all: they are
downloaded separately from `Qwen/Qwen3.5-2B-Base`. What this repo ships is the difference
between that base model and a decision model.
""")

code('''
for name in sorted(os.listdir(checkpoint)):
    kind = "dir " if os.path.isdir(os.path.join(checkpoint, name)) else "file"
    print(f"{kind}  {name}")
''')

md("""
Three files carry the model. `hobson_config.json` is the architecture and calibration
record, `lora/` is the adapter, and `head.safetensors` is the readout.

The config name is historical. The loader looks for `strands_decider_config.json` first and
falls back to `hobson_config.json`, which is what checkpoints published before the rename
carry. Read it rather than inferring anything from the model card.
""")

code('''
config = json.load(open(os.path.join(checkpoint, "hobson_config.json")))
for key in ("base_model", "head_type", "pointer_dim", "num_slots", "max_length",
            "torch_dtype", "use_lora", "lora_r", "lora_alpha",
            "temperature", "temperature_by_kind", "ordinal_smoothing"):
    print(f"{key:20} {config[key]}")
''')

md("""
`head_type` is the line that matters: `pointer`, not `slot`. Everything in section 4
follows from it.

`num_slots: 24` is vestigial here. It sizes the fixed-width readout that earlier versions
used, and a pointer head has no per-slot parameters, so nothing in this checkpoint is
capped at 24 options. `evaluate` in `strands_decider/infer.py` enforces the ceiling only
when `head_type != "pointer"`.

Now the adapter. LoRA trains two small matrices per targeted projection and leaves the
base weights untouched, so the adapter's size is the honest measure of how much of this
model is new.
""")

code('''
adapter = json.load(open(os.path.join(checkpoint, "lora", "adapter_config.json")))
print("base model     ", adapter["base_model_name_or_path"])
print("rank           ", adapter["r"], " alpha", adapter["lora_alpha"])
print("target modules ", ", ".join(sorted(adapter["target_modules"])))

with safe_open(os.path.join(checkpoint, "lora", "adapter_model.safetensors"), "pt") as f:
    shapes = {k: f.get_slice(k).get_shape() for k in f.keys()}
lora_params = sum(torch.Size(s).numel() for s in shapes.values())
print(f"\\n{len(shapes)} adapter tensors, {lora_params:,} parameters")
''')

md("""
The target list names both attention families. `q_proj`, `k_proj`, `v_proj` and `o_proj`
are the softmax-attention layers; `in_proj_a`, `in_proj_b`, `in_proj_qkv`, `in_proj_z` and
`out_proj` are the gated-delta linear-attention layers. The adapter reaches every layer of
the hybrid, not just the conventional ones.

The readout is the other trained part, and it is tiny.
""")

code('''
head_state = load_file(os.path.join(checkpoint, "head.safetensors"))
for key, value in head_state.items():
    print(f"{key:12} {tuple(value.shape)}  {value.dtype}")

head_params = sum(v.numel() for v in head_state.values())
print(f"\\nhead  {head_params:,} parameters")
print(f"total trained  {lora_params + head_params:,}")
''')

md("""
Two projections, `q` and `k`, each 2048 to 256, plus a LayerNorm over the 2048-wide hidden
state. There is no output layer of width 24, or of any fixed width. That absence is the
architecture.

The head is stored in float32 while the torso is bfloat16. A readout is small enough that
the precision costs nothing, and a low-precision classifier is a needless source of
calibration error.
""")

md("""
## 3. The prompt is the API

Nothing in the weights knows what a department is. The prompt establishes, per request,
which option is in which position, and the readout is generic over those positions. That
binding is the whole reason the model answers label sets it never saw in training.

`render_question` does the binding and is worth reading directly, because its output is
the actual input to the model.
""")

code('''
from strands_decider.prompting import (read_choice, read_noul, read_score,
                                       render_question, render_state)
from strands_decider.schema import SystemOneRequest

MESSAGE = "When I stop at lights the car pulls hard to the left and the pedal sinks almost to the floor."

request = SystemOneRequest(
    state=MESSAGE,
    questions={
        "department": {
            "type": "choice",
            "instructions": "Which dealership department should handle this?",
            "criteria": {
                "Service": "Diagnose, repair, or maintain the vehicle.",
                "Parts": "Sell parts the customer takes away.",
                "Sales": "Buy, lease, or trade in a vehicle.",
                "Warranty Claims": "Refunds or disputes about coverage, no repair visit.",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How soon does the vehicle need attention?",
            "criteria": ["Routine", "Within a week", "Immediately"],
        },
        "safety_risk": {"type": "noul", "instructions": "Is it unsafe to keep driving this vehicle?"},
    },
)

rendered = {name: render_question(q) for name, q in request.questions.items()}
print(render_state(request.state))
print(rendered["department"].text)
''')

md("""
The state is a separate block, rendered once. Each question is a suffix that ends at
`<answer>`, and the options are numbered from 1 so that position k in the list is
unambiguous in the text itself.

All three question kinds render as the same numbered list. A yes/no question is a
two-option list whose labels are `false` and `true`; a score is a list whose labels are the
level indices.
""")

code('''
for name, rq in rendered.items():
    print(f"{name}  kind={rq.kind}  slots={rq.n_slots}  labels={rq.slot_labels}")

print()
print(rendered["safety_risk"].text)
print()
print(rendered["urgency"].text)
''')

md("""
This is why a `noul` is not a special case in the model. It is a choice over two options,
and `read_noul` recovers the answer by looking up which position `true` landed in, so a
shuffled rendering cannot flip the answer.

Each rendered question also records the character span of every option's line. A pointer
head needs these: it scores option k from the last token inside span k.
""")

code('''
rq = rendered["department"]
for i, (start, end) in enumerate(rq.option_spans):
    print(f"slot {i}  {rq.slot_labels[i]:16} span=({start},{end})  {rq.text[start:end]!r}")
''')

md("""
Descriptions are collapsed to a single line on purpose. One option is always exactly one
line, which is what keeps the span unambiguous.

Because the heads are generic over position, training has to shuffle option order. A
positional head that always sees the positive class at slot 0 memorises that instead of
reading the option. The pointer head makes genericity structural rather than something the
shuffling has to enforce, which is the argument for it.

## 4. The pointer head

Load the model now. This brings down the base decoder on first use.
""")

code('''
from strands_decider.infer import load_engine

engine = load_engine(MODEL, device="cpu")
model = engine.model

print("head        ", type(model.head).__name__)
print("hidden size ", model.head.q.in_features)
print("pointer dim ", model.head.q.out_features)
print("hybrid torso", type(model).is_hybrid(model.torso))
_, total = model.trainable_parameters()
print(f"parameters   {total:,} total, {lora_params + head_params:,} of them trained")
''')

md("""
`PointerHead.forward` is four lines, and they are a single attention score:

```python
def forward(self, decide, options):           # decide [B, d], options [B, K, d]
    d = self.q(self.dropout(self.norm(decide))).unsqueeze(-1)   # query  [B, dim, 1]
    o = self.k(self.dropout(self.norm(options)))                # keys   [B, K, dim]
    return (o @ d).squeeze(-1) * self.scale                     # logits [B, K]
```

The query comes from the `<answer>` position, which under causal attention has read the
state and the whole question. Each key comes from the last token of one option's line,
which has read that option. The logit for option k is the dot product of the two, scaled
by `dim ** -0.5`.

Three things follow, and they are the reasons to prefer this over a fixed-width head.
The option count is unbounded, because there is no output dimension to exceed. An option's
logit depends on what the option says rather than on where it sits, so there is no
per-slot parameter for a positional bias to live in. And the parameter count does not grow
with the number of options.

The package reports a measurement for the trade: on frozen features, +0.063 accuracy over
a slot head refitted on the same rows at `pointer_dim=256`, and +0.058 at
`pointer_dim=16`, where a 2.1M-parameter MLP slot head reaches only +0.032. The gain is
the addressing, not the capacity.

## 5. One instrumented forward pass

Now take the pieces apart and rebuild one answer by hand. The point is to see that nothing
is hidden: the published probability is a dot product, a division and a softmax.
""")

code('''
from strands_decider.modeling import (apply_temperature, gather_options,
                                      masked_log_softmax, pool_last_token)

state_text = render_state(request.state)
prefix_ids, suffix_ids = engine._fit(state_text, [rq.text])
print(f"state tokens    {len(prefix_ids)}")
print(f"question tokens {len(suffix_ids[0])}")

ids, mask = engine._pad([prefix_ids + suffix_ids[0]])
option_idx = engine._option_idx([rq], len(prefix_ids))
print(f"option token positions {option_idx.tolist()[0]}")
''')

md("""
The question is tokenised before the state, not after. `_fit` gives the question first
claim on the context window, because the options and the `<answer>` marker are what make
a task answerable at all, while the state is the part that can be trimmed. Truncating from
the right, as a naive concatenation would, cuts away the very positions the head reads.

One forward pass through the torso, with the language-model head never loaded.
""")

code('''
with torch.inference_mode():
    hidden = model.encode(input_ids=ids, attention_mask=mask)

pooled = pool_last_token(hidden, mask).to(torch.float32)
options = gather_options(hidden, option_idx).to(torch.float32)

print(f"hidden   {tuple(hidden.shape)}")
print(f"pooled   {tuple(pooled.shape)}   <- the query position")
print(f"options  {tuple(options.shape)}  <- one key per option")

print(f"\\nquery reads token {model.tokenizer.decode(ids[0, -1])!r}")
for i, pos in enumerate(option_idx[0].tolist()):
    print(f"key {i} reads token {model.tokenizer.decode(ids[0, pos])!r} "
          f"at position {pos}, end of {rq.slot_labels[i]!r}")
''')

md("""
The query sits on the final token of `<answer>`. Each key sits on the last token of its
option line, which for these options is the full stop that ends the description. The
model is not reading the label; it is reading the position that has just finished reading
the label and its description.

Four options, four keys, one query. Score them.
""")

code('''
with torch.inference_mode():
    raw = model.head(pooled, options)

for label, logit in zip(rq.slot_labels, raw[0].tolist()):
    print(f"{label:16} {logit:+.4f}")
''')

md("""
These are raw logits, and they are not yet the answer. Two steps remain, and both are
visible in the config rather than buried in the weights.
""")

code('''
temperature = engine._temperatures([rq.kind])
print(f"kind={rq.kind}  temperature={temperature.item():.4f}")

logits = apply_temperature(raw, temperature)
log_probs = masked_log_softmax(logits, torch.tensor([rq.n_slots]))
probs = log_probs.exp()[0, : rq.n_slots].tolist()

by_hand = read_choice(probs, rq)
print("\\nby hand:", {k: round(v, 4) for k, v in by_hand.items()})

from_engine = engine.ask(request.state, {"department": request.questions["department"]})
print("engine: ", from_engine.answers["department"].probabilities)
''')

md("""
The two agree. The answer the model publishes is the softmax of one scaled dot product per
option, divided by a constant from a JSON file.

`masked_log_softmax` is the step that lets a single head serve a two-option `noul` and a
ten-level `score` without the unused positions stealing probability mass. It softmaxes over
each row's first `n_slots` entries only. For the pointer head there are exactly as many
keys as options, so the mask binds only when a batch mixes questions of different widths,
which is the normal case: one request can carry all three primitives.

## 6. Three temperatures, one per primitive

Calibration is where this checkpoint differs from Decider-4b in a way that shows up in the
numbers rather than the architecture. Decider-4b has one scalar, 1.935, applied to every
question. This model fits one temperature per primitive.
""")

code('''
print(f"fallback  {config['temperature']:.4f}")
for kind, value in config["temperature_by_kind"].items():
    direction = "sharpens" if value < 1 else "softens"
    print(f"{kind:8}  {value:.4f}  {direction}")
''')

md("""
Below 1 the distribution sharpens, above 1 it flattens. `choice` at 0.734 is sharpened,
`score` at 1.328 is flattened, and `noul` at 0.911 is close to untouched. A single scalar
cannot do this: the three primitives sit at different accuracies, so a temperature that
calibrates one over-softens another.

Hold the logits fixed and vary only the temperature, to see what the constant is worth.
""")

code('''
for t in (1.0, config["temperature_by_kind"]["choice"], config["temperature_by_kind"]["score"]):
    p = masked_log_softmax(apply_temperature(raw, torch.tensor([t])),
                           torch.tensor([rq.n_slots])).exp()[0]
    top = p.max().item()
    print(f"T={t:.3f}  top={top:.4f}  " + "  ".join(f"{v:.3f}" for v in p.tolist()))
''')

md("""
Same forward pass, same ranking, different confidence. That is all a temperature can do,
and it is why the model card reports Brier score and expected calibration error next to
accuracy rather than accuracy alone.

The limitation is stated plainly in the model card and is worth repeating: these three
temperatures were fitted on held-out short classification tasks only. On a rubric unlike
the ones it was fitted on, the ranking may survive while the confidence does not.

Run all three questions together to see the per-primitive temperatures used in one request.
""")

code('''
answers = engine.ask(request.state, request.questions).answers

print(f"department   {answers['department'].choice} "
      f"(confidence {answers['department'].confidence:.2f})")
urgency = answers["urgency"]
print(f"urgency      {urgency.legend[str(round(urgency.score))]} (score {urgency.score:.2f})")
print(f"safety_risk  {answers['safety_risk'].noul:.2f}")
''')

md("""
One request, three questions, one shared state. The state is encoded once and the three
question suffixes run against that cache, so the second and third questions cost only
their own tokens.

Note the `urgency` answer. A score is not a classification of the top level: it is the
expected level index under the distribution, so it can land between rubric levels, and
`legend` records what each level meant. Its confidence is ordinal rather than
max-probability, because being one level off is a smaller error than being three off.

## 7. Where the three read paths diverge

All three models in this repo's experiments take the same typed question and return a
calibrated distribution over the options. They disagree about where the distribution comes
from.

| | Laya | Decider-4b | strands-decider-2B |
|---|---|---|---|
| Direction | Bidirectional encoder | Causal decoder | Causal decoder |
| Read position | Pooled sequence representation | The next-token slot after `<answer>` | One position per option, plus `<answer>` |
| Read mechanism | A classification head | The LM head, restricted to option-label tokens | Dot product of a query and per-option keys |
| Options the readout can address | Fixed by the head's width | Fixed by the prompt's label tokens | Unbounded |
| Positional bias possible | Yes, per class | Yes, per label token | No per-option parameter exists |
| Calibration | One temperature per kind | One scalar, 1.935 | One temperature per kind |

The middle column is the one that gets over-generalised. "A decoder reads the decision
straight out of its next-token distribution, restricted to the option tokens" is a correct
description of Decider-4b and a wrong description of this model, which never computes a
next-token distribution at all. Both are decoders; only one uses the language-model head.

## 8. What the published numbers say

The checkpoint ships its own evaluation, which is worth reading from the files rather than
the card.
""")

code('''
summary = json.load(open(os.path.join(checkpoint, "eval", "summary.json")))
print("sections:", ", ".join(summary))

for name, block in summary.items():
    if name == "internal":
        continue
    print(f"\\n{name}")
    print(json.dumps(block, indent=2)[:600])
''')

md("""
On JevBench public at a 4096-token window: 167 of 231 correct, accuracy 0.723, Brier
0.348, expected calibration error 0.050. The 3072-token window scores the same accuracy
with slightly worse calibration, which matches the config comment that the context curve
goes flat past 3072.

The internal results spread wide by task shape, and the spread is the useful part. Many
of the generated buckets hold a handful of items, so filter by `n` before reading anything
into the extremes.
""")

code('''
internal = summary["internal"]
rows = sorted((kv for kv in internal.items() if kv[1]["n"] >= 100),
              key=lambda kv: kv[1]["accuracy"])
for name, block in rows:
    print(f"{block['accuracy']:.3f}  n={block['n']:>5}  {name}")
''')

md("""
The model card names the weak spots directly, and they are consistent with the spread:
questions are read less carefully than documents, long multi-step documents are the
weakest case, and `score` and `noul` transfer poorly to rubrics unlike the training ones.
It was trained on public datasets, so it inherits their label noise.

The training cost is in the checkpoint too: one p5.48xlarge, 4204 seconds of recipe wall
clock, of which 1685 seconds was the training stage. Seventeen point nine million trained
parameters on a frozen 2B torso is a cheap artefact, which is the practical argument for
this shape of model.

### What this means for steering an agent

`steering_demo.py` in this directory uses this model as the sixth steering method, against
the same policy and the same two questions as the other five. Running it surfaces
something a model card will not tell you. On the demo scenario above, the department is
right with probability 0.97 and `safety_risk` comes back at 0.57, just under the 0.6
threshold the policy uses, so the agent books standard priority where same-day is correct.

Across all 50 scenarios in `scenarios.jsonl` it scores 72 percent, and the miss set is the
interesting part. Eleven of its fourteen misses are unsafe vehicles scored just below the
threshold, in the 0.5 to 0.6 band. One is a false alarm just above it, at 0.65 on a recall
notice with no warning lights. Only two are genuine routing errors. So nearly the whole
error budget sits inside a tenth of a probability unit around one constant.

That is the honest shape of the trade. A 2B model with a 17.9M-parameter diff answers in
about three seconds on a CPU and costs nothing per call, and on hard items it lands near
the boundary rather than away from it. Two things follow for anyone using it this way: the
threshold is a parameter you own and should fit on your own labelled scenarios rather than
inherit, and the model card's advice to re-fit the per-primitive temperatures on your own
rubric is the thing to act on before trusting a boundary at all.

### Where to look next

- `model_architecture.ipynb` in this directory, for Laya and Decider-4b side by side.
- `strands_decider_basics.py`, the smallest complete call.
- `strands_decider/prompting.py` in the installed package. Its module docstring is the
  clearest statement of why the heads are generic.
- `strands_decider/modeling.py`, for `PointerHead` and `SlotHead` in the same file, which
  is the cleanest way to see what the pointer head removes.
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

out = Path(__file__).with_name("strands_decider_architecture.ipynb")
out.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"{out.name}: {len(cells)} cells")
