"""Run the Laya model locally and triage a car dealership message.

Laya is an open-weight System 1 decision model on Hugging Face. It accepts the
same typed questions as Jev. Laya classifies. Plain Python decides.
The first run downloads about 800 MB of weights.
"""

import os
import warnings

os.environ.setdefault("USE_TF", "0")  # avoid slow TensorFlow import in transformers
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", module="laya")  # checkpoint calibration notice

import laya  # noqa: E402

MODEL = "convaiinnovations/laya"

MESSAGE = (
    "My 2024 sedan's brakes started grinding this morning and the pedal "
    "feels soft. It's still under warranty. Can I bring it in today?"
)

agent = laya.load(MODEL)

result = agent.predict(
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

print(f"department:  {department} ({answers['department']['answer_confidence']:.2f})")
print(f"urgency:     {urgency['legend'][str(round(urgency['score']))]} ({urgency['score']:.2f})")
print(f"safety_risk: {safety_risk:.2f}")

priority = safety_risk >= 0.6 or urgency["score"] >= 1.5
print(f"action:      {'same-day priority booking' if priority else 'standard booking'} ({department})")
