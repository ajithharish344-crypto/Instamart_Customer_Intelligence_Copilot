"""Run the main Instamart analytics pipeline from the project root."""
from src.data_pipeline import audit, clean_data, validate
from src.customer_models import rfm_and_segments
from src.lapse_model_improved import build_lapse_model
from src.triage import run_pipeline, FEW
from src.policy_rag import PolicyRAG, evaluate
def main():
    print("\n=== PART A: AUDIT / CLEAN / VALIDATE ===")
    audit()
    clean_data()
    validate()
    print("\n=== PART B: CUSTOMER SEGMENTS ===")
    rfm, diagnostics, profile, pca = rfm_and_segments(k=4)
    print(diagnostics.to_string(index=False))
    print(profile.to_string(index=False))
    print("PCA explained variance:", pca)
    print("\n=== PART B: IMPROVED LAPSE MODEL ===")
    print(build_lapse_model())
    print("\n=== PART C: TICKET TRIAGE ===")
    triage, summary = run_pipeline(prompt=FEW, token_budget=100000)
    print(summary)
    print(triage.head().to_string(index=False))
    print("\n=== PART D: POLICY RAG EVALUATION ===")
    rag = PolicyRAG()
    print("Total policy chunks:", len(rag.chunks))
    print(evaluate(rag).to_string(index=False))
    print("\nMAIN PIPELINE COMPLETED.")
    print("Review data/, outputs/, and data/instamart.db.")
    print("Run the agent scenario and checkpoint tests separately.")
if __name__ == "__main__":
    main()
