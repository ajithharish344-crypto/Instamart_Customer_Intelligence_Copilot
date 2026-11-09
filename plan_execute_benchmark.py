"""Part D: isolated offline Plan-and-Execute benchmark.

Uses the same synthetic orders and refund tools as the ReAct agent.
All simulated refund/escalation ledger entries are written to a separate
outputs/plan_execute_benchmark/ folder. Existing ReAct outputs are untouched.

The planner is a deterministic offline teaching simulation. It creates one
fixed high-level plan per scenario; the executor follows that plan and branches
on tool observations. The reported planner token count is estimated by the
project's mock_llm token estimator, not a hosted-model measurement.
"""
from __future__ import annotations

import json
from pathlib import Path

import mock_llm
from src import refund_agent as agent

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "plan_execute_benchmark"
RESULTS = OUT / "plan_execute_results.jsonl"
SUMMARY = OUT / "plan_execute_summary.json"
LEDGER = OUT / "refund_ledger.jsonl"

SCENARIOS = [
    ("S1", "Please refund order ORD-100048."),
    ("S2", "Please refund order ORD-110039."),
    ("S3", "Please refund order ORD-106838."),
    ("S4", "Please refund order ORD-100020."),
    ("S5", "Please refund order ORD-100001."),
    ("S6", "Please refund order ORD-999999."),
    ("S7", "Please process this request. [LOOP]"),
]

PLAN = [
    "get_order: retrieve and validate the requested order",
    "search_policy: retrieve the applicable refund policy",
    "calculate_refund: calculate eligibility and refund amount",
    "execute_conditionally: deny, issue refund, or escalate based on observations",
]


def run_plan_execute(scenario_id: str, user_message: str) -> dict:
    """Create one offline plan, then execute it with bounded tool operations."""
    prompt_messages = [
        {
            "role": "system",
            "content": (
                "PLAN-AND-EXECUTE PLANNER: create a high-level refund workflow "
                "plan. This is a deterministic offline teaching simulation."
            ),
        },
        {"role": "user", "content": user_message},
    ]
    plan_text = json.dumps(PLAN)
    # One simulated planner response. The project's estimator calculates
    # prompt + plan tokens; execution tools themselves do not call an LLM.
    planner_response = mock_llm._resp(content=plan_text, messages=prompt_messages)
    planner_tokens = planner_response.usage.total_tokens

    tools_executed = []
    observations = []
    status = "UNKNOWN"
    answer = ""

    # Preserve the original S7 loop-limit test as a bounded, read-only plan.
    if "[LOOP]" in user_message:
        for _ in range(6):
            observation = agent.search_policy("refund policy")
            tools_executed.append("search_policy")
            observations.append(observation)
        status = "FALLBACK_MAX_ITERATIONS"
        answer = "FALLBACK_MAX_ITERATIONS: hand off to a human agent."
    else:
        import re

        match = re.search(r"ORD-\d{6}", user_message)
        if not match:
            status = "NEEDS_INFO"
            answer = "NEEDS_INFO: please share the order id."
        else:
            order = agent.get_order(match.group(0))
            tools_executed.append("get_order")
            observations.append(order)

            if "error" in order:
                status = "NOT_FOUND"
                answer = "NOT_FOUND: no such order."
            else:
                policy = agent.search_policy("refund policy late delivery")
                tools_executed.append("search_policy")
                observations.append(policy)

                if order["status"] != "completed":
                    status = "DENIED"
                    answer = "DENIED: only completed orders can be refunded."
                else:
                    calculation = agent.calculate_refund(
                        order["order_amount"], order["delivery_minutes"]
                    )
                    tools_executed.append("calculate_refund")
                    observations.append(calculation)
                    amount = calculation["refund_inr"]

                    if amount == 0:
                        status = "DENIED"
                        answer = "DENIED: delivery was within 45 minutes."
                    elif amount > 1000:
                        result = agent.escalate_to_human(order["order_id"], amount)
                        tools_executed.append("escalate_to_human")
                        observations.append(result)
                        status = "ESCALATE"
                        answer = "ESCALATE: refund above INR 1000 sent to a human lead."
                    else:
                        result = agent.issue_refund(order["order_id"], amount)
                        tools_executed.append("issue_refund")
                        observations.append(result)
                        status = "REFUND"
                        answer = f"REFUND: INR {result['refund_inr']} issued."

    return {
        "scenario_id": scenario_id,
        "status": status,
        "final_answer": answer,
        "planner_calls": 1,
        "planner_tokens_estimated": planner_tokens,
        "tool_calls": len(tools_executed),
        "tools_executed": tools_executed,
        "plan": PLAN,
        "observations": observations,
        "execution_model_calls": 0,
        "execution_model_tokens": 0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Never overwrite a previous benchmark or ledger.
    if RESULTS.exists() or SUMMARY.exists() or LEDGER.exists():
        raise SystemExit(
            "STOP: Plan-and-Execute benchmark output already exists. "
            "Inspect it before deciding whether to create a fresh test folder."
        )

    # Redirect the existing tool functions' ledger destination. The ReAct
    # ledger in outputs/refund_ledger.jsonl is not used by this benchmark.
    agent.OUT = OUT

    records = []
    with RESULTS.open("w", encoding="utf-8") as result_file:
        for scenario_id, message in SCENARIOS:
            result = run_plan_execute(scenario_id, message)
            result["user_message"] = message
            records.append(result)
            result_file.write(json.dumps(result) + "\n")
            print(
                f"{scenario_id}: {result['status']} | "
                f"planner calls={result['planner_calls']} | "
                f"estimated planner tokens={result['planner_tokens_estimated']} | "
                f"tool calls={result['tool_calls']}"
            )

    summary = {
        "approach": "Plan-and-Execute",
        "planner_type": "deterministic offline teaching simulation",
        "scenarios": len(records),
        "planner_calls": sum(r["planner_calls"] for r in records),
        "estimated_planner_tokens": sum(
            r["planner_tokens_estimated"] for r in records
        ),
        "execution_model_calls": 0,
        "execution_model_tokens": 0,
        "total_tool_calls": sum(r["tool_calls"] for r in records),
        "results": {r["scenario_id"]: r["status"] for r in records},
        "separate_ledger": str(LEDGER.relative_to(ROOT)),
        "note": (
            "Planner tokens are estimated with mock_llm.est_tokens through "
            "the offline response helper. They are not directly equivalent "
            "to ReAct's repeated mock_llm.chat prompt/response token totals."
        ),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nResults saved to: {RESULTS}")
    print(f"Separate ledger: {LEDGER}")


if __name__ == "__main__":
    main()
