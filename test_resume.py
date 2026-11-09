from pathlib import Path
import src.refund_agent as agent
run_id = "resume_checkpoint_demo"
state_file = Path("state") / f"{run_id}.json"
test_ledger_dir = Path("outputs") / "resume_test"
if state_file.exists() or (test_ledger_dir / "refund_ledger.jsonl").exists():
    raise SystemExit("STOP: Test files already exist. Do not repeat this test.")
# Redirect only the test ledger. The main ledger stays untouched.
agent.OUT = test_ledger_dir
print("STEP 1: Simulating a crash after two tool executions...")
try:
    agent.run_agent(
        "Please refund order ORD-100048.",
        scenario_id="resume_checkpoint_demo",
        run_id=run_id,
        crash_after=2
    )
except RuntimeError as error:
    print("Simulated crash:", error)
print("Checkpoint exists:", state_file.exists())
print("\nSTEP 2: Resuming from the saved checkpoint...")
result = agent.resume_agent(run_id)
print("\n=== RESUME TEST RESULT ===")
print("Status:", result["status"])
print("Answer:", result["final_answer"])
print("Model calls:", result["llm_calls"])
print("Tools executed:", [step["tool"] for step in result["steps"]])
print("Tokens:", result["tokens_used"])
ledger = test_ledger_dir / "refund_ledger.jsonl"
print("Separate test ledger exists:", ledger.exists())
if ledger.exists():
    print("Test ledger entries:", len(ledger.read_text(encoding="utf-8").splitlines()))
expected_tools = [
    "get_order",
    "search_policy",
    "calculate_refund",
    "issue_refund"
]
assert result["status"] == "REFUND"
assert result["final_answer"] == "REFUND: INR 128.25 issued."
assert [step["tool"] for step in result["steps"]] == expected_tools
assert ledger.exists()
assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1
print("\nCHECKPOINT AND RESUME TEST PASSED")
