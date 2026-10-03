"""Measure accuracy and latency of the steering methods.

Each scenario in scenarios.jsonl holds a customer message and the booking that
steering should produce (department and priority). A scenario is correct when
the agent's final booking matches both.

Usage: uv run python benchmark.py [deterministic|llm|jev|laya|decider|strands ...] [--limit N] [--workers N]
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import steering_demo

steering_demo.log = lambda *args: None

SCENARIOS = Path(__file__).with_name("scenarios.jsonl")


def run_scenario(method: str, scenario: dict) -> tuple[bool, float, bool]:
    """Run one scenario end to end. Return (correct, latency in seconds, errored)."""
    agent = steering_demo.build_agent(method)
    start = time.perf_counter()
    errored = False
    try:
        agent(scenario["message"])
    except Exception:
        errored = True  # a failed run counts as incorrect
    latency = time.perf_counter() - start
    booking = agent.state.get("booking") or {}
    expected = {"department": scenario["expected_department"], "priority": scenario["expected_priority"]}
    return booking == expected, latency, errored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("methods", nargs="*", default=list(steering_demo.HANDLERS))
    parser.add_argument("--limit", type=int, help="run only the first N scenarios")
    parser.add_argument("--workers", type=int, default=8, help="scenarios to run in parallel")
    args = parser.parse_args()

    scenarios = [json.loads(line) for line in SCENARIOS.read_text().splitlines() if line.strip()]
    scenarios = scenarios[: args.limit]

    for method, load in (
        ("laya", steering_demo.laya_agent),
        ("decider", steering_demo.decider_agent),
        ("strands", steering_demo.strands_agent),
    ):
        if method in args.methods:
            load()  # load the local model before timing starts

    print(f"{'method':<14}{'accuracy':>10}{'total (s)':>12}{'avg (s)':>10}{'errors':>8}  missed ids")
    for method in args.methods:
        start = time.perf_counter()
        with ThreadPoolExecutor(args.workers) as pool:
            results = list(pool.map(lambda s: run_scenario(method, s), scenarios))
        total = time.perf_counter() - start

        correct = [ok for ok, _, _ in results]
        accuracy = sum(correct) / len(results)
        average = sum(latency for _, latency, _ in results) / len(results)
        errors = sum(errored for _, _, errored in results)
        missed = [s["id"] for s, ok in zip(scenarios, correct) if not ok]
        print(f"{method:<14}{accuracy:>10.0%}{total:>12.1f}{average:>10.1f}{errors:>8}  {missed}")


if __name__ == "__main__":
    main()
