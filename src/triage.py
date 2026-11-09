
"""Part C: Instamart customer support ticket triage."""

import json
import random
import re
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import mock_llm


ROOT = Path(__file__).resolve().parents[1]
TICKETS_FILE = ROOT / "data" / "raw" / "tickets.csv"
SEGMENTS_FILE = ROOT / "outputs" / "customer_segments.csv"
OUTPUT_DIR = ROOT / "outputs"

ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}
LONG_TOKEN_LIMIT = 400
DEFAULT_BUDGET = 100000

ZERO_SHOT = """You are an Instamart customer support ticket classifier.
CLASSIFY each customer message into exactly one category:
billing, delivery, technical, or fraud.
Return JSON only, using this exact structure:
{"category":"one allowed category","reason":"short reason"}
Do not include Markdown or any text outside the JSON."""

FEW = """You are an Instamart customer support ticket classifier.
CLASSIFY each customer message into exactly one category:
billing, delivery, technical, or fraud.
Return JSON only, using this exact structure:
{"category":"one allowed category","reason":"short reason"}
Do not include Markdown or any text outside the JSON.

EXAMPLES:
Customer: I was charged twice for my order.
Output: {"category":"billing","reason":"duplicate charge"}

Customer: My delivery rider has not arrived.
Output: {"category":"delivery","reason":"delivery delay"}

Customer: Someone hacked my account and placed an order.
Output: {"category":"fraud","reason":"unauthorized account activity"}

Customer: The app crashes when I try to log in.
Output: {"category":"technical","reason":"app login issue"}"""


def redact_pii(text):
    """Redact email addresses, Indian mobile numbers, and order IDs."""
    text = str(text)

    counts = {"emails": 0, "phones": 0, "order_ids": 0}

    patterns = [
        ("emails", r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]"),
        ("phones", r"\b[6-9]\d{9}\b", "[PHONE]"),
        ("order_ids", r"ORD-\d{6}", "[ORDER_ID]"),
    ]

    for key, pattern, replacement in patterns:
        text, count = re.subn(pattern, replacement, text)
        counts[key] += count

    # Confirm that none of the original PII patterns remain.
    assert not re.search(patterns[0][1], text)
    assert not re.search(patterns[1][1], text)
    assert not re.search(patterns[2][1], text)

    return text, counts


def estimate_tokens(text):
    return mock_llm.est_tokens(text)


def validate_response(response_text):
    """Validate JSON output, then try the permitted keyword fallback."""
    try:
        data = json.loads(response_text)
        category = str(data.get("category", "")).lower().strip()

        if category in ALLOWED_CATEGORIES:
            return category, str(data.get("reason", "")), "classified_json"

    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    words = re.findall(r"[a-z]+", (response_text or "").lower())
    found = ALLOWED_CATEGORIES.intersection(words)

    if len(found) == 1:
        category = next(iter(found))
        return category, "Recovered from output keyword", "classified_keyword"

    return None, "Model output could not be validated", "flagged_invalid_output"


def call_classifier(text, system_prompt, ticket_id, retry_log):
    """Call the mock model with up to three retries."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    for attempt in range(4):
        try:
            response = mock_llm.chat(messages)
            message = response.choices[0].message
            usage = response.usage.total_tokens

            return message.content, usage, attempt

        except mock_llm.TransientLLMError:
            if attempt >= 3:
                raise

            wait = 0.05 * (2 ** attempt) * random.uniform(0.9, 1.15)

            retry_log.append({
                "ticket_id": ticket_id,
                "retry_number": attempt + 1,
                "wait_seconds": round(wait, 4),
            })

            time.sleep(wait)

    raise RuntimeError("Maximum retry attempts reached")


def load_tickets():
    if not TICKETS_FILE.exists():
        raise FileNotFoundError(f"Ticket file not found: {TICKETS_FILE}")

    df = pd.read_csv(TICKETS_FILE)

    required = {"ticket_id", "customer_id", "order_id", "text", "true_category"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Ticket dataset is missing columns: {missing}")

    assert len(df) == 164
    return df


def classify_run(df, system_prompt, token_budget=DEFAULT_BUDGET):
    """Run one complete classification pass."""
    mock_llm.reset()
    random.seed(42)

    results = []
    retry_log = []
    total_tokens = 0
    pii_totals = {"emails": 0, "phones": 0, "order_ids": 0}

    for _, row in df.iterrows():
        ticket_id = row["ticket_id"]
        original_text = row["text"]

        base = {
            "ticket_id": ticket_id,
            "customer_id": row["customer_id"],
            "true_category": row["true_category"],
            "category": None,
            "reason": "",
            "status": "",
            "tokens_used": 0,
            "retry_count": 0,
        }

        # Empty tickets are skipped without a model call.
        if pd.isna(original_text) or not str(original_text).strip():
            base["status"] = "skipped_empty"
            results.append(base)
            continue

        redacted_text, pii_counts = redact_pii(original_text)

        for key in pii_totals:
            pii_totals[key] += pii_counts[key]

        # Long tickets are flagged without a model call.
        estimated = estimate_tokens(redacted_text)

        if estimated > LONG_TOKEN_LIMIT:
            base["status"] = "flagged_long"
            base["tokens_used"] = estimated
            results.append(base)
            continue

        # Check the budget before each model call.
        if total_tokens >= token_budget:
            base["status"] = "budget_exhausted"
            results.append(base)
            continue

        try:
            output, used, retries = call_classifier(
                redacted_text, system_prompt, ticket_id, retry_log
            )
            total_tokens += used

            category, reason, status = validate_response(output)

            base.update({
                "category": category,
                "reason": reason,
                "status": status,
                "tokens_used": used,
                "retry_count": retries,
            })

        except mock_llm.TransientLLMError:
            base["status"] = "flagged_model_error"

        results.append(base)

    result_df = pd.DataFrame(results)

    # No ticket text or PII is written to the result CSV.
    assert "text" not in result_df.columns

    summary = {
        "total_tickets": len(df),
        "total_tokens": total_tokens,
        "retry_errors_recovered": len(retry_log),
        "retry_log": retry_log,
        "pii_redaction_counts": pii_totals,
        "status_counts": result_df["status"].value_counts().to_dict(),
    }

    return result_df, summary


def score_results(result_df):
    """Calculate classification metrics on successfully classified tickets."""
    scored = result_df[
        result_df["status"].isin(["classified_json", "classified_keyword"])
        & result_df["true_category"].isin(ALLOWED_CATEGORIES)
    ].copy()

    if scored.empty:
        return {
            "classified_count": 0,
            "accuracy": 0.0,
            "confusion_matrix": [],
            "classification_report": {},
        }

    labels = ["billing", "delivery", "technical", "fraud"]

    return {
        "classified_count": len(scored),
        "accuracy": accuracy_score(scored["true_category"], scored["category"]),
        "confusion_matrix": confusion_matrix(
            scored["true_category"], scored["category"], labels=labels
        ).tolist(),
        "classification_report": classification_report(
            scored["true_category"],
            scored["category"],
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def add_customer_routing(result_df):
    """Add customer segment, priority, and support route."""
    if not SEGMENTS_FILE.exists():
        raise FileNotFoundError(
            "Customer segments file missing. Run Part B first."
        )

    segments = pd.read_csv(SEGMENTS_FILE)
    segment_lookup = segments.set_index("customer_id")["segment_name"].to_dict()

    result = result_df.copy()
    result["segment_name"] = result["customer_id"].map(segment_lookup).fillna("Unknown")
    result["priority"] = "normal"
    result["route"] = "support_queue"

    for index, row in result.iterrows():
        if row["status"] not in ("classified_json", "classified_keyword"):
            result.at[index, "route"] = "human_review"

        elif row["category"] == "fraud":
            result.at[index, "priority"] = "high"
            result.at[index, "route"] = "security_team"

        elif row["segment_name"] == "Premium":
            result.at[index, "priority"] = "high"
            result.at[index, "route"] = "premium_desk"

    return result


def run_pipeline(prompt=FEW, token_budget=DEFAULT_BUDGET):
    """Run zero-shot, few-shot, and budget-demo experiments."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_tickets()

    print("\n=== ZERO-SHOT CLASSIFICATION ===")
    zero_df, zero_summary = classify_run(df, ZERO_SHOT, DEFAULT_BUDGET)
    zero_metrics = score_results(zero_df)

    print("Zero-shot status counts:", zero_summary["status_counts"])
    print("Zero-shot accuracy:", round(zero_metrics["accuracy"], 4))
    print("Zero-shot tokens:", zero_summary["total_tokens"])
    print("Zero-shot confusion matrix:")
    print(zero_metrics["confusion_matrix"])

    print("\n=== FEW-SHOT CLASSIFICATION ===")
    few_df, few_summary = classify_run(df, FEW, DEFAULT_BUDGET)
    few_metrics = score_results(few_df)

    print("Few-shot status counts:", few_summary["status_counts"])
    print("Few-shot accuracy:", round(few_metrics["accuracy"], 4))
    print("Few-shot tokens:", few_summary["total_tokens"])

    print("\n=== TOKEN BUDGET DEMONSTRATION ===")
    budget_df, budget_summary = classify_run(df, ZERO_SHOT, 6000)
    print("Budget demo status counts:", budget_summary["status_counts"])
    assert (budget_df["status"] == "budget_exhausted").any()

    # Route the final few-shot results.
    final_df = add_customer_routing(few_df)

    final_columns = [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used",
    ]

    final_df[final_columns].to_csv(
        OUTPUT_DIR / "triage_results.csv", index=False
    )

    budget_df.to_csv(
        OUTPUT_DIR / "triage_budget_demo.csv", index=False
    )

    # Save retry evidence for both classification experiments.
    retry_rows = []

    for mode, summary in [
        ("zero_shot", zero_summary),
        ("few_shot", few_summary),
        ("budget_demo", budget_summary),
    ]:
        for item in summary["retry_log"]:
            retry_rows.append({"mode": mode, **item})

    pd.DataFrame(retry_rows).to_csv(
        OUTPUT_DIR / "triage_retry_log.csv", index=False
    )

    comparison = {
        "zero_shot": {
            "accuracy": zero_metrics["accuracy"],
            "classified_count": zero_metrics["classified_count"],
            "total_tokens": zero_summary["total_tokens"],
            "tokens_per_classified_ticket": (
                zero_summary["total_tokens"] / zero_metrics["classified_count"]
                if zero_metrics["classified_count"] else 0
            ),
            "confusion_matrix": zero_metrics["confusion_matrix"],
            "status_counts": zero_summary["status_counts"],
            "retry_errors_recovered": zero_summary["retry_errors_recovered"],
        },
        "few_shot": {
            "accuracy": few_metrics["accuracy"],
            "classified_count": few_metrics["classified_count"],
            "total_tokens": few_summary["total_tokens"],
            "tokens_per_classified_ticket": (
                few_summary["total_tokens"] / few_metrics["classified_count"]
                if few_metrics["classified_count"] else 0
            ),
            "confusion_matrix": few_metrics["confusion_matrix"],
            "status_counts": few_summary["status_counts"],
            "retry_errors_recovered": few_summary["retry_errors_recovered"],
        },
        "pii_redaction_counts": few_summary["pii_redaction_counts"],
        "budget_demo_status_counts": budget_summary["status_counts"],
    }

    with open(OUTPUT_DIR / "triage_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n=== PII REDACTION COUNTS ===")
    print(few_summary["pii_redaction_counts"])

    print("\n=== FINAL FEW-SHOT ROUTING COUNTS ===")
    print(final_df["route"].value_counts().to_dict())

    print("\nSaved:")
    print("outputs/triage_results.csv")
    print("outputs/triage_budget_demo.csv")
    print("outputs/triage_retry_log.csv")
    print("outputs/triage_comparison.json")

    summary = {
        "zero_shot": zero_metrics,
        "few_shot": few_metrics,
        "zero_shot_tokens": zero_summary["total_tokens"],
        "few_shot_tokens": few_summary["total_tokens"],
        "pii_redaction_counts": few_summary["pii_redaction_counts"],
        "zero_shot_status_counts": zero_summary["status_counts"],
        "few_shot_status_counts": few_summary["status_counts"],
        "budget_status_counts": budget_summary["status_counts"],
    }

    return final_df, summary
