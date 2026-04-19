import os
import joblib
import pandas as pd
from src.agent import run_agentic_audit
from dotenv import load_dotenv

load_dotenv()


def test_e2e_scenarios():
    print("🚀 Starting End-to-End Agent Verification...\n")

    # Load model
    model_path = os.path.join("models", "credit_risk_model_v2.pkl")
    if not os.path.exists(model_path):
        print("❌ Model file not found. Run training first.")
        return
    model = joblib.load(model_path)

    # Scenario 1: High Risk (Large amount, short duration, low savings)
    high_risk_client = {
        "Age": 22,
        "Sex": "male",
        "Job": 1,
        "Housing": "rent",
        "Saving accounts": "little",
        "Checking account": "little",
        "Credit amount": 15000,
        "Duration": 48,
        "Purpose": "car",
    }

    # Scenario 2: Low Risk (Mature, own house, high savings)
    low_risk_client = {
        "Age": 45,
        "Sex": "female",
        "Job": 2,
        "Housing": "own",
        "Saving accounts": "rich",
        "Checking account": "rich",
        "Credit amount": 2000,
        "Duration": 12,
        "Purpose": "education",
    }

    scenarios = [
        ("High Risk - Large Amount", high_risk_client),
        ("Low Risk - Standard", low_risk_client),
    ]

    for label, client in scenarios:
        print(f"--- 🧪 Testing Scenario: {label} ---")
        try:
            result = run_agentic_audit(model, client)

            print(f"✅ Workflow Status: {result['status']}")
            print(
                f"✅ ML Prediction: {result['ml_result']['risk_label']} ({result['ml_result']['default_probability']}%)"
            )
            print(f"✅ RAG Context Length: {len(result['policy_context'])} chars")
            print(
                f"✅ Auditor reasoning provided: {'Yes' if result['audit_summary'] else 'No'}"
            )
            print("\n--- 🤖 Auditor Reasoning (Excerpt) ---")
            print(result["audit_summary"][:300] + "...")
            print("-" * 50 + "\n")

        except Exception as e:
            print(f"❌ Error in scenario {label}: {str(e)}")


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️ GROQ_API_KEY not found. Ensure it is set in .env or environment.")
    else:
        test_e2e_scenarios()
