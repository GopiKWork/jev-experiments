"""Run the Strands decision model locally and triage a car dealership message.

Strands Decider is an open-weight System 1 decision model on Hugging Face. It is
a LoRA adapter on the Qwen3.5-2B-Base decoder with the language-model head
removed, plus a small pointer head that scores each option from that option's own
hidden state. It accepts the same typed questions as Jev, Laya and Decider.
The model classifies. Plain Python decides.

The first run downloads about 4.5 GB: the base decoder, plus the adapter and
head from the checkpoint repo.
"""

import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Qwen3.5-2B-Base is a hybrid: some layers use linear attention. If the
# flash-linear-attention package imports, transformers routes those layers through a
# Triton kernel that needs a GPU driver. Hiding the package selects the reference
# PyTorch path instead, which is slower but runs on CPU. Delete this line on a GPU.
sys.modules["fla"] = None

from strands_decider.infer import load_engine  # noqa: E402

MODEL = "StrandsAgents/strands-decider-2B-hobson-v19"
DEVICE = "cpu"

MESSAGE = (
    "My 2024 sedan's brakes started grinding this morning and the pedal "
    "feels soft. It's still under warranty. Can I bring it in today?"
)

engine = load_engine(MODEL, device=DEVICE)

result = engine.ask(
    MESSAGE,
    {
        # Choice: pick one label from a set. Descriptions are required here.
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
        # Score: place the text on an ordered rubric, lowest level first.
        "urgency": {
            "type": "score",
            "instructions": "How soon does the vehicle need attention?",
            "criteria": ["Routine", "Within a week", "Immediately"],
        },
        # Noul: probability that the answer is yes.
        "safety_risk": {"type": "noul", "instructions": "Is it unsafe to keep driving this vehicle?"},
    },
)

answers = result.answers
department = answers["department"]
urgency = answers["urgency"]
safety_risk = answers["safety_risk"].noul

print(f"department:  {department.choice} ({department.confidence:.2f})")
print(f"urgency:     {urgency.legend[str(round(urgency.score))]} ({urgency.score:.2f})")
print(f"safety_risk: {safety_risk:.2f}")

priority = safety_risk >= 0.6 or urgency.score >= 1.5
print(f"action:      {'same-day priority booking' if priority else 'standard booking'} ({department.choice})")
