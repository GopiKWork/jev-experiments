"""Call the Jev model through OpenRouter and triage a car dealership message.

Jev classifies the message. Plain Python decides what to do with it.
"""

import os

from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

load_dotenv()

MODEL = "~typesafe/jev-latest"
BASE_URL = "https://openrouter.ai/api"

MESSAGE = (
    "My 2024 sedan's brakes started grinding this morning and the pedal "
    "feels soft. It's still under warranty. Can I bring it in today?"
)

client = TypeSafeClient(api_key=os.environ["OPENROUTER_API_KEY"], base_url=BASE_URL)

response = client.system_one(
    model=MODEL,
    state=MESSAGE,
    questions={
        # Choice: pick one label from a set.
        "department": Choice(
            instructions="Which dealership department should handle this?",
            criteria={
                "Service": None,
                "Parts": None,
                "Sales": None,
                "Warranty Claims": None,
            },
        ),
        # Score: place the text on an ordered rubric.
        "urgency": Score(
            instructions="How soon does the vehicle need attention?",
            criteria=["Routine", "Within a week", "Immediately"],
        ),
        # Noul: probability that the answer is yes.
        "safety_risk": Noul(instructions="Is it unsafe to keep driving this vehicle?"),
    },
)

answers = response.answers
department = answers["department"].choice
urgency = answers["urgency"]
safety_risk = answers["safety_risk"].noul

print(f"department:  {department} ({answers['department'].confidence:.2f})")
print(f"urgency:     {urgency.legend[round(urgency.score)]} ({urgency.score:.2f})")
print(f"safety_risk: {safety_risk:.2f}")

priority = safety_risk >= 0.6 or urgency.score >= 1.5
print(f"action:      {'same-day priority booking' if priority else 'standard booking'} ({department})")
