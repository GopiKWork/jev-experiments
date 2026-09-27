"""Ask the same three questions about one customer message, on all three models.

The message is the one the blog uses throughout. It contains no safety keyword, so a
keyword rule misses it. This script is the recorded source for the per-answer numbers
the blog quotes.

    uv run python running_example.py [jev|laya|decider|all]
"""

import os
import sys
import warnings

from dotenv import load_dotenv

load_dotenv()

JEV_MODEL = "~typesafe/jev-latest"
BASE_URL = "https://openrouter.ai/api"
LAYA_MODEL = "convaiinnovations/laya"
DECIDER_MODEL = "Mapika/decider-4b"
DECIDER_REVISION = "v2"

STATE = "When I stop at lights the car pulls hard to the left and the pedal sinks almost to the floor."

DEPARTMENTS = ["Service", "Parts", "Sales", "Warranty Claims"]
LEVELS = ["Routine", "Within a week", "Immediately"]
DEPARTMENT_Q = "Which dealership department should handle this?"
URGENCY_Q = "How soon does this need attention?"
SAFETY_Q = "Is it unsafe to keep driving this vehicle?"

SCHEMA = {
    "department": {"type": "choice", "instructions": DEPARTMENT_Q, "criteria": {d: None for d in DEPARTMENTS}},
    "urgency": {"type": "score", "instructions": URGENCY_Q, "criteria": LEVELS},
    "safety_risk": {"type": "noul", "instructions": SAFETY_Q},
}


def report(name, department, confidence, urgency, safety_risk, extra=""):
    print(f"[{name}] department={department} ({confidence}) urgency={urgency} safety_risk={safety_risk}{extra}")


def run_jev():
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

    client = TypeSafeClient(api_key=os.environ["OPENROUTER_API_KEY"], base_url=BASE_URL)
    answers = client.system_one(
        model=JEV_MODEL,
        state=STATE,
        questions={
            "department": Choice(instructions=DEPARTMENT_Q, criteria={d: None for d in DEPARTMENTS}),
            "urgency": Score(instructions=URGENCY_Q, criteria=LEVELS),
            "safety_risk": Noul(instructions=SAFETY_Q),
        },
    ).answers
    urgency = answers["urgency"]
    report(
        "jev",
        answers["department"].choice,
        round(answers["department"].confidence, 4),
        f"{urgency.score} {urgency.legend[round(urgency.score)]!r}",
        round(answers["safety_risk"].noul, 4),
        f" urgency_confidence={round(urgency.confidence, 4)}",
    )
    print(f"[jev] probabilities {answers['department'].probabilities}")
    print(f"[jev] urgency probabilities {urgency.probabilities}")


def run_laya():
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    warnings.filterwarnings("ignore", module="laya")  # checkpoint calibration notice
    import laya

    answers = laya.load(LAYA_MODEL).predict(STATE, SCHEMA)["answers"]
    urgency = answers["urgency"]
    report(
        "laya",
        answers["department"]["choice"],
        answers["department"]["answer_confidence"],
        f"{urgency['score']} {urgency['legend'][str(round(urgency['score']))]!r}",
        answers["safety_risk"]["noul"],
    )
    print(f"[laya] probabilities {answers['department']['probabilities']}")
    print(f"[laya] urgency probabilities {urgency['probabilities']}")


def run_decider():
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    sys.modules.setdefault("fla", None)  # its Triton kernel needs a GPU, so use the torch path
    from decider.infer import Decider
    from huggingface_hub import snapshot_download

    d = Decider(snapshot_download(DECIDER_MODEL, revision=DECIDER_REVISION))
    answers = d.system_one(STATE, SCHEMA)["answers"]
    urgency = answers["urgency"]
    report(
        "decider",
        answers["department"]["choice"],
        answers["department"]["confidence"],
        f"{urgency['score']} {urgency['legend'][str(round(urgency['score']))]!r}",
        answers["safety_risk"]["noul"],
        f" fit_mass={urgency['fit_mass']}",
    )
    print(f"[decider] probabilities {answers['department']['probabilities']}")
    print(f"[decider] urgency probabilities {urgency['probabilities']}")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in ["jev", "laya", "decider"] if which == "all" else [which]:
        {"jev": run_jev, "laya": run_laya, "decider": run_decider}[name]()


if __name__ == "__main__":
    main()
