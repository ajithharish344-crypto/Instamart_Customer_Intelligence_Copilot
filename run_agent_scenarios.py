import json
from pathlib import Path
from src.refund_agent import run_agent
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
RESULT_FILE = OUT / "agent_runs.jsonl"
LEDGER_FILE = OUT / "refund_ledger.jsonl"
if RESULT_FILE.exists():
    raise SystemExit(
        "STOP: agent_runs.jsonl already exists. Do not rerun and duplicate transactions."
    )
before = len(LEDGER_FILE.read_text(encoding="utf-8").splitlines()) if LEDGER_FILE.exists() else 0
scenarios = [
    ("S1", "Please refund order ORD-100048."),
    ("S2", "Please refund order ORD-110039."),
    ("S3", "Please refund order ORD-106838."),
    ("S4", "Please refund order ORD-100020."),
    ("S5", "Please refund order ORD-100001."),
    ("S6", "Please refund order ORD-999999."),
    ("S7", "Please process this request. [LOOP]"),
]
results = []
for scenario_id, message in scenarios:
    print(f"\nRunning {scenario_id}...")
    result = run_agent(message, scenario_id=scenario_id)
    record = {"scenario_id": scenario_id, **result}
    results.append(record)
    with RESULT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print("Status:", result["status"])
    print("Model calls:", result["llm_calls"])
    print("Tokens:", result["tokens_used"])
    print("Tools:", [s["tool"] for s in result["steps"]])
    print("Answer:", result["final_answer"])
after = len(LEDGER_FILE.read_text(encoding="utf-8").splitlines()) if LEDGER_FILE.exists() else 0
print("\n=== SCENARIO SUMMARY ===")
for r in results:
    print(r["scenario_id"], "->", r["status"])
print("\nNew ledger entries:", after - before)
print("Agent results saved:", RESULT_FILE)
