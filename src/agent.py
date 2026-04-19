# ABOUTME: Implements the Agentic AI workflow using LangGraph.
# ABOUTME: Coordinates between the XGBoost ML model, RAG policy retrieval, and Groq/Llama3 LLM.

import os
import pandas as pd
from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv

from src.predict import make_prediction
from src.rag import get_retriever

load_dotenv()


# ── State Definition ──────────────────────────────────────────────────────────
class AgentState(TypedDict):
    client_data: dict  # Input raw data
    input_df: pd.DataFrame  # Processed df for model
    ml_result: dict  # Result from XGBoost
    policy_context: str  # Context retrieved from RAG
    audit_summary: str  # Final LLM reasoning
    messages: List[BaseMessage]
    status: str  # Current workflow status
    model: object  # ML Model object


# ── Nodes ─────────────────────────────────────────────────────────────────────


def ml_inference_node(state: AgentState):
    """Call the traditional XGBoost model."""
    model = state.get("model")  # Passed in from app
    input_df = state["input_df"]

    result = make_prediction(model, input_df)

    return {"ml_result": result, "status": "ML_INFERRED"}


def policy_retrieval_node(state: AgentState):
    """Retrieve relevant snippets from the credit policy knowledge base."""
    retriever = get_retriever()
    client_profile = state["client_data"]

    context = retriever.get_policy_context(client_profile)

    return {"policy_context": context, "status": "POLICY_RETRIEVED"}


def auditor_node(state: AgentState):
    """LLM Auditor that combines ML predictions with Policy rules."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    ml = state["ml_result"]
    client = state["client_data"]
    policy = state["policy_context"]

    prompt = f"""
You are the "CreditRisk AI Auditor", a senior financial risk agent.
Your task is to provide a detailed audit of a loan application.

### CLIENT PROFILE:
{client}

### MACHINE LEARNING PREDICTION:
- Prediction: {ml['risk_label']}
- Default Probability: {ml['default_probability']}%
- Confidence: {ml['confidence']}%

### RELEVANT BANK POLICIES (RAG):
{policy}

### YOUR TASK:
1. Review the ML prediction against the Bank Policy.
2. Check for "Hard Overrides" (e.g., loan amount limits, age constraints).
3. If the ML model and Policy conflict, prioritize the Policy.
4. Provide a "Final Recommendation" (Approve/Decline).
5. Explain the reasoning clearly for a junior banker.

Format your response in Markdown.
"""

    response = llm.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content="Generate Agentic Audit Report."),
        ]
    )

    return {"audit_summary": response.content, "status": "AUDITED"}


# ── Workflow Construction ─────────────────────────────────────────────────────


def create_agent_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("predict_risk", ml_inference_node)
    workflow.add_node("retrieve_policy", policy_retrieval_node)
    workflow.add_node("generate_audit", auditor_node)

    workflow.set_entry_point("predict_risk")

    workflow.add_edge("predict_risk", "retrieve_policy")
    workflow.add_edge("retrieve_policy", "generate_audit")
    workflow.add_edge("generate_audit", END)

    return workflow.compile()


# Global instance
_agent = None


def get_credit_agent():
    global _agent
    if _agent is None:
        _agent = create_agent_workflow()
    return _agent


def run_agentic_audit(model, client_data: dict):
    """Entry point for the Streamlit app to run the full agentic flow.

    Accepts either:
      - Original column-name keys: {'Age': 22, 'Sex': 'male', ...}
      - Snake_case keys from app.py: {'age': 22, 'sex': 'male', ...}
    """
    agent = get_credit_agent()

    # ── Normalize keys to match the model's trained column names ─────────────
    KEY_MAP = {
        "age": "Age",
        "sex": "Sex",
        "job": "Job",
        "housing": "Housing",
        "saving_accounts": "Saving accounts",
        "checking_account": "Checking account",
        "credit_amount": "Credit amount",
        "duration": "Duration",
        "purpose": "Purpose",
    }
    normalized = {KEY_MAP.get(k, k): v for k, v in client_data.items()}

    # Prep initial state
    input_df = pd.DataFrame([normalized])

    initial_state = {
        "client_data": normalized,
        "input_df": input_df,
        "model": model,  # Hidden dependency for the ml_inference_node
        "status": "STARTING",
    }

    return agent.invoke(initial_state)
