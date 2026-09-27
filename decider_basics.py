"""Run the Decider model locally and triage a car dealership message.

Decider is an open-weight System 1 decision model on Hugging Face. It is a
Qwen3 decoder with the generation loop removed: the model reads the hidden state
at one answer slot per question and scores only the option labels. It accepts
the same typed questions as Jev and Laya. Decider classifies. Plain Python
decides. The first run downloads about 8.4 GB of weights.
"""

import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Decider-4b is a Qwen3.5 hybrid: 24 of its 32 layers use linear attention. If the
# flash-linear-attention package imports, transformers routes those layers through a
# Triton kernel that needs a GPU driver. Hiding the package selects the reference
# PyTorch path instead, which is slower but runs on CPU. Delete this line on a GPU.
sys.modules["fla"] = None

from decider.infer import Decider  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

MODEL = "Mapika/decider-4b"
REVISION = "v2"  # better calibrated on hard items than v2.1 on the main branch

MESSAGE = (
    "My 2024 sedan's brakes started grinding this morning and the pedal "
    "feels soft. It's still under warranty. Can I bring it in today?"
)

# Pin the revision so a later release does not change the result.
decider = Decider(snapshot_download(MODEL, revision=REVISION))

result = decider.system_one(
    MESSAGE,
    {
        # Choice: pick one label from a set.
        "department": {
            "type": "choice",
            "instructions": "Which dealership department should handle this?",
            "criteria": {"Service": None, "Parts": None, "Sales": None, "Warranty Claims": None},
        },
        # Score: place the text on an ordered rubric.
        "urgency": {
            "type": "score",
            "instructions": "How soon does the vehicle need attention?",
            "criteria": ["Routine", "Within a week", "Immediately"],
        },
        # Noul: probability that the answer is yes.
        "safety_risk": {"type": "noul", "instructions": "Is it unsafe to keep driving this vehicle?"},
    },
)

answers = result["answers"]
department = answers["department"]["choice"]
urgency = answers["urgency"]
safety_risk = answers["safety_risk"]["noul"]

print(f"department:  {department} ({answers['department']['confidence']:.2f})")
print(f"urgency:     {urgency['legend'][str(round(urgency['score']))]} ({urgency['score']:.2f})")
print(f"safety_risk: {safety_risk:.2f}")

priority = safety_risk >= 0.6 or urgency["score"] >= 1.5
print(f"action:      {'same-day priority booking' if priority else 'standard booking'} ({department})")
