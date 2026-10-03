"""Three ways to steer a Strands agent before it calls a tool.

A dealership agent books appointments. Its system prompt tells it to book
everything as standard priority. A steering handler checks each booking
against the dealership policy and sends the agent back with guidance when the
department or priority is wrong.

    deterministic  Python keyword rules decide.
    llm            A second LLM reads the policy and decides.
    jev            Jev classifies the message. Python decides.
    laya           Laya, an open-weight model run locally, classifies. Python decides.
    decider        Decider, a larger open-weight model run locally, classifies. Python decides.
    strands        Strands Decider, an open-weight model run locally, classifies. Python decides.

Usage: uv run python steering_demo.py [deterministic|llm|jev|laya|decider|strands]
"""

import os
import sys
import threading
import warnings
from functools import cache

from dotenv import load_dotenv
from strands import Agent, ToolContext, tool
from strands.models.bedrock import BedrockModel
from strands.vended_plugins.steering import Guide, LLMSteeringHandler, Proceed, SteeringHandler
from typesafe_sdk import Choice, Noul, TypeSafeClient

load_dotenv()

OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
JEV_MODEL = "~typesafe/jev-latest"
AGENT_MODEL = "us.openai.gpt-5.6-terra"  # OpenAI model on Amazon Bedrock
LAYA_MODEL = "convaiinnovations/laya"  # open weights on Hugging Face, run locally
DECIDER_MODEL = "Mapika/decider-4b"  # open weights on Hugging Face, run locally
DECIDER_REVISION = "v2"  # better calibrated on hard items than v2.1 on the main branch
STRANDS_MODEL = "StrandsAgents/strands-decider-2B-hobson-v19"  # open weights, run locally

# No word like "brake" or "smoke" appears, so keyword rules miss it.
CUSTOMER = "When I stop at lights the car pulls hard to the left and the pedal sinks almost to the floor."

DEPARTMENTS = {
    "Service": "Diagnose, repair, or maintain the customer's vehicle.",
    "Parts": "Sell parts or accessories that the customer takes away or installs elsewhere.",
    "Sales": "Buy, lease, or trade in a vehicle.",
    "Warranty Claims": "Refunds, reimbursements, or disputes about warranty coverage. No repair visit.",
}

POLICY = (
    "Route each booking to one department: "
    + " ".join(f"{name}: {desc}" for name, desc in DEPARTMENTS.items())
    + " Use priority 'same-day' only for a Service booking where the vehicle is unsafe to drive."
    " Use priority 'standard' for everything else."
)

AGENT_PROMPT = (
    "You book dealership appointments. Book every request with book_service right away, "
    "without asking the customer for more details. Always use priority 'standard' unless a "
    "tool result tells you otherwise, then follow it. Reply to the customer in one sentence."
)

model = BedrockModel(model_id=AGENT_MODEL, region_name="us-east-1", max_tokens=1024)
jev = TypeSafeClient(api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api")

log = print  # benchmark.py replaces this to silence output


@tool(context=True)
def book_service(department: str, priority: str, notes: str, tool_context: ToolContext) -> str:
    """Book a dealership appointment.

    Args:
        department: One of Service, Parts, Sales, Warranty Claims.
        priority: Either "standard" or "same-day".
        notes: Short summary of the customer's issue.
    """
    tool_context.agent.state.set("booking", {"department": department, "priority": priority})
    log(f"[tool]  booked {department} / {priority}")
    return f"Booked {department} appointment, priority {priority}."


def show(action):
    """Print the steering decision and pass it through."""
    log(f"[steer] {type(action).__name__}: {action.reason[:100]}")
    return action


def customer_text(agent: Agent) -> str:
    """Return the first user message, which is the customer's request."""
    return agent.messages[0]["content"][0]["text"]


def check(booking: dict, department: str, same_day: bool):
    """Compare the proposed booking with the expected one. Shared by the code-based methods."""
    priority = "same-day" if same_day else "standard"
    if booking["department"] != department or booking["priority"] != priority:
        return show(Guide(reason=f"Book department '{department}' with priority '{priority}'."))
    return show(Proceed(reason="booking matches policy"))


# Method 1: deterministic code. Fast and predictable, but only as good as the keyword lists.
class KeywordSteering(SteeringHandler):
    SAFETY_WORDS = ("brake", "smoke", "airbag", "steering", "fire")
    DEPARTMENT_WORDS = {
        "Warranty Claims": ("warranty", "reimburse", "refund", "claim"),
        "Sales": ("trade in", "lease", "test drive", "new model", "pre-owned"),
        "Parts": ("order", "in stock", "pick them up", "install it myself", "shipped"),
    }

    async def steer_before_tool(self, *, agent, tool_use, **kwargs):
        text = customer_text(agent).lower()
        department = next(
            (name for name, words in self.DEPARTMENT_WORDS.items() if any(w in text for w in words)),
            "Service",
        )
        same_day = department == "Service" and any(w in text for w in self.SAFETY_WORDS)
        return check(tool_use["input"], department, same_day)


# Method 2: LLM steering. Understands language, but is slower and its decision is opaque.
class LLMSteering(LLMSteeringHandler):
    def __init__(self):
        super().__init__(system_prompt=f"You review dealership bookings. {POLICY}", model=model)

    async def steer_before_tool(self, **kwargs):
        return show(await super().steer_before_tool(**kwargs))


# Method 3: Jev steering. Jev returns probabilities. Plain Python owns the policy.
class JevSteering(SteeringHandler):
    THRESHOLD = 0.6

    async def steer_before_tool(self, *, agent, tool_use, **kwargs):
        answers = jev.system_one(
            model=JEV_MODEL,
            state=customer_text(agent),
            questions={
                "safety_risk": Noul(instructions="Is it unsafe to keep driving this vehicle?"),
                "department": Choice(
                    instructions="Which dealership department should handle this?",
                    criteria=DEPARTMENTS,
                ),
            },
        ).answers
        safety_risk = answers["safety_risk"].noul
        department = answers["department"].choice
        log(f"[jev]   safety_risk={safety_risk:.2f} department={department}")

        same_day = department == "Service" and safety_risk >= self.THRESHOLD
        return check(tool_use["input"], department, same_day)


@cache
def laya_agent():
    """Load Laya once, on first use. The first run downloads about 800 MB of weights."""
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    warnings.filterwarnings("ignore", module="laya")  # checkpoint calibration notice
    import laya

    return laya.load(LAYA_MODEL)


# Method 4: Laya steering. Same questions and policy as Jev, but the model runs on this machine.
class LayaSteering(SteeringHandler):
    THRESHOLD = 0.6
    lock = threading.Lock()  # one local model, so run one prediction at a time

    async def steer_before_tool(self, *, agent, tool_use, **kwargs):
        with self.lock:
            answers = laya_agent().predict(
                customer_text(agent),
                {
                    "safety_risk": {"type": "noul", "instructions": "Is it unsafe to keep driving this vehicle?"},
                    "department": {
                        "type": "choice",
                        "instructions": "Which dealership department should handle this?",
                        "criteria": DEPARTMENTS,
                    },
                },
            )["answers"]
        safety_risk = answers["safety_risk"]["noul"]
        department = answers["department"]["choice"]
        log(f"[laya]  safety_risk={safety_risk:.2f} department={department}")

        same_day = department == "Service" and safety_risk >= self.THRESHOLD
        return check(tool_use["input"], department, same_day)


@cache
def decider_agent():
    """Load Decider once, on first use. The first run downloads about 8.4 GB of weights."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    sys.modules.setdefault("fla", None)  # its Triton kernel needs a GPU, so use the torch path
    from decider.infer import Decider
    from huggingface_hub import snapshot_download

    return Decider(snapshot_download(DECIDER_MODEL, revision=DECIDER_REVISION))


# Method 5: Decider steering. Same questions and policy again, on a 4.2B local model.
class DeciderSteering(SteeringHandler):
    THRESHOLD = 0.6
    lock = threading.Lock()  # one local model, so run one prediction at a time

    async def steer_before_tool(self, *, agent, tool_use, **kwargs):
        with self.lock:
            answers = decider_agent().system_one(
                customer_text(agent),
                {
                    "safety_risk": {"type": "noul", "instructions": "Is it unsafe to keep driving this vehicle?"},
                    "department": {
                        "type": "choice",
                        "instructions": "Which dealership department should handle this?",
                        "criteria": DEPARTMENTS,
                    },
                },
            )["answers"]
        safety_risk = answers["safety_risk"]["noul"]
        department = answers["department"]["choice"]
        log(f"[decider] safety_risk={safety_risk:.2f} department={department}")

        same_day = department == "Service" and safety_risk >= self.THRESHOLD
        return check(tool_use["input"], department, same_day)


@cache
def strands_agent():
    """Load Strands Decider once, on first use. The first run downloads about 4.5 GB."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    sys.modules.setdefault("fla", None)  # its Triton kernel needs a GPU, so use the torch path
    from strands_decider.infer import load_engine

    return load_engine(STRANDS_MODEL, device="cpu")


# Method 6: Strands Decider steering. Same questions and policy, on a 2B local model with
# a pointer head. Its choice options carry descriptions, which this readout scores directly.
class StrandsDeciderSteering(SteeringHandler):
    THRESHOLD = 0.6
    lock = threading.Lock()  # one local model, so run one prediction at a time

    async def steer_before_tool(self, *, agent, tool_use, **kwargs):
        with self.lock:
            answers = strands_agent().ask(
                customer_text(agent),
                {
                    "safety_risk": {"type": "noul", "instructions": "Is it unsafe to keep driving this vehicle?"},
                    "department": {
                        "type": "choice",
                        "instructions": "Which dealership department should handle this?",
                        "criteria": DEPARTMENTS,
                    },
                },
            ).answers
        safety_risk = answers["safety_risk"].noul
        department = answers["department"].choice
        log(f"[strands] safety_risk={safety_risk:.2f} department={department}")

        same_day = department == "Service" and safety_risk >= self.THRESHOLD
        return check(tool_use["input"], department, same_day)


HANDLERS = {
    "deterministic": KeywordSteering,
    "llm": LLMSteering,
    "jev": JevSteering,
    "laya": LayaSteering,
    "decider": DeciderSteering,
    "strands": StrandsDeciderSteering,
}


def build_agent(method: str) -> Agent:
    return Agent(
        model=model,
        tools=[book_service],
        plugins=[HANDLERS[method]()],
        system_prompt=AGENT_PROMPT,
        callback_handler=None,
    )


def main() -> None:
    method = sys.argv[1] if len(sys.argv) > 1 else "jev"
    log(f"[method] {method}")
    result = build_agent(method)(CUSTOMER)
    log(f"[agent] {str(result).strip()}")


if __name__ == "__main__":
    main()
