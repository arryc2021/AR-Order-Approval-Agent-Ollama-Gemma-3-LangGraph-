# ar_agent.py
# AR Order Approval Agent — Ollama (Gemma 3) + LangGraph, file-based, no DB.

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, Literal, List

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

# LangChain (Ollama chat LLM) + LangGraph
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --------------------------------------------------------------------------------------
# Logging setup (two files: workflow + llm interactions)
# --------------------------------------------------------------------------------------

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
WF_LOG_PATH = f"ar_workflow_{TIMESTAMP}.log"
LLM_LOG_PATH = "llm_interactions.log"

os.makedirs(".", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(WF_LOG_PATH), logging.StreamHandler()],
)

llm_logger = logging.getLogger("llm")
_llm_fh = logging.FileHandler(LLM_LOG_PATH)
_llm_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
llm_logger.setLevel(logging.INFO)
llm_logger.addHandler(_llm_fh)

# --------------------------------------------------------------------------------------
# Types / State
# --------------------------------------------------------------------------------------

ApprovalStatus = Literal["approved", "rejected", "escalate"]

class ARState(BaseModel):
    # inputs
    customer: Dict[str, Any] = Field(default_factory=dict)
    order: Dict[str, Any] = Field(default_factory=dict)
    policy_text: str = ""

    # processing artifacts
    credit_assessment: Optional[Dict[str, Any]] = None
    approval_status: Optional[ApprovalStatus] = None
    decision_reason: Optional[str] = None

    # meta
    order_id: Optional[str] = None
    customer_id: Optional[str] = None
    human_note: Optional[str] = None  # filled if escalated

# --------------------------------------------------------------------------------------
# Helpers / IO
# --------------------------------------------------------------------------------------

REQUIRED_CUSTOMER_COLUMNS = {"customer_id", "credit_limit", "outstanding_balance"}
REQUIRED_ORDER_COLUMNS = {"order_id", "customer_id", "order_amount"}

def require_columns(df: pd.DataFrame, required: set, name: str):
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")

def load_policy_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_customer_map(path: str) -> Dict[str, Dict[str, Any]]:
    df = pd.read_csv(path, dtype=str)
    require_columns(df, REQUIRED_CUSTOMER_COLUMNS, "customer_master.csv")
    # Cast numerics if present
    for col in ("credit_limit", "outstanding_balance"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Normalize ID
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    return {row["customer_id"]: row._asdict() if hasattr(row, "_asdict") else row.to_dict()
            for _, row in df.iterrows()}

def load_orders(path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(path, dtype=str)
    require_columns(df, REQUIRED_ORDER_COLUMNS, "sales_order.csv")
    # Cast numerics if present
    df["order_amount"] = pd.to_numeric(df["order_amount"], errors="coerce")
    # Normalize IDs
    df["order_id"] = df["order_id"].astype(str).str.strip()
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    return [row._asdict() if hasattr(row, "_asdict") else row.to_dict()
            for _, row in df.iterrows()]

def write_decisions_csv(rows: List[Dict[str, Any]], out_path: Optional[str] = None) -> str:
    out_path = out_path or f"ar_decisions_{TIMESTAMP}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path

def ensure_arstate(obj) -> ARState:
    """Normalize LangGraph return (may be dict) to ARState."""
    if isinstance(obj, ARState):
        return obj
    if isinstance(obj, dict):
        try:
            return ARState(**obj)
        except ValidationError as e:
            # Be resilient if state schema changes mid-run
            logging.warning("Coercing dict -> ARState failed: %s", e)
            # keep known keys only
            allowed = {k: obj.get(k) for k in ARState.model_fields.keys()}
            return ARState(**allowed)
    raise TypeError(f"Unexpected state type: {type(obj)}")

# --------------------------------------------------------------------------------------
# LLM: Ollama + Gemma 3
# --------------------------------------------------------------------------------------

def make_llm(model_name: str = "gemma3:latest", temperature: float = 0.2) -> ChatOllama:
    # Uses local Ollama server at http://localhost:11434
    return ChatOllama(model=model_name, temperature=temperature, num_ctx=8192)

def assess_credit_with_llm(
    llm: ChatOllama,
    policy_text: str,
    customer: Dict[str, Any],
    order: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Ask Gemma 3 for a structured credit assessment & decision proposal.
    Returns strict JSON or a safe 'escalate' fallback if parsing fails.
    """
    system = SystemMessage(content=(
        "You are a credit analyst for Accounts Receivable. "
        "Follow the company's credit policy precisely. "
        "Return STRICT JSON with keys: "
        "decision (approved|rejected|escalate), "
        "reason (string), "
        "policy_matches (array of strings), "
        "risk_flags (array of strings), "
        "computed (object with available_credit:number, would_exceed:boolean)."
    ))
    human = HumanMessage(content=(
        f"Company Credit Policy:\n```\n{policy_text}\n```\n\n"
        f"Customer Master (JSON):\n```json\n{json.dumps(customer, indent=2)}\n```\n\n"
        f"Sales Order (JSON):\n```json\n{json.dumps(order, indent=2)}\n```\n\n"
        "Decide approve, reject, or escalate for human review. "
        "Respond with STRICT JSON only, no prose."
    ))

    resp = llm.invoke([system, human])  # sync call
    content = (resp.content or "").strip()

    # Log prompt/response (masking emails lightly)
    safe_prompt = human.content.replace("@", " at ")
    llm_logger.info("PROMPT >>>\n%s", safe_prompt)
    llm_logger.info("RESPONSE <<<\n%s", content)

    # Parse JSON robustly
    try:
        text = content
        if text.startswith("```"):
            # strip fencing
            text = text.strip("`")
            # remove possible language header
            if "\n" in text:
                text = text[text.find("\n")+1:]
        data = json.loads(text)
        # sanitize decision
        dec = str(data.get("decision", "escalate")).lower().strip()
        if dec not in {"approved", "rejected", "escalate"}:
            data["decision"] = "escalate"
            data["reason"] = f"{data.get('reason','')} (Unrecognized decision; escalated.)"
        return data
    except Exception as e:
        return {
            "decision": "escalate",
            "reason": f"LLM output could not be parsed as JSON: {e}. Raw: {content[:280]}...",
            "policy_matches": [],
            "risk_flags": ["parse_error"],
            "computed": {},
        }

# --------------------------------------------------------------------------------------
# Graph nodes
# --------------------------------------------------------------------------------------

def node_prepare(state: ARState) -> ARState:
    logging.info("[prepare] order_id=%s customer_id=%s",
                 state.order.get("order_id"), state.order.get("customer_id"))
    state.order_id = str(state.order.get("order_id", "")).strip()
    state.customer_id = str(
        state.order.get("customer_id", state.customer.get("customer_id", ""))
    ).strip()
    return state

def node_llm_assess(state: ARState, llm: ChatOllama) -> ARState:
    logging.info("[llm_assess] order_id=%s → Gemma 3", state.order_id)
    state.credit_assessment = assess_credit_with_llm(llm, state.policy_text, state.customer, state.order)
    return state

def node_decide(state: ARState) -> ARState:
    logging.info("[decide] order_id=%s", state.order_id)
    assessment = state.credit_assessment or {}
    decision = str(assessment.get("decision", "escalate")).lower().strip()
    reason = assessment.get("reason", "No reason provided by LLM.")

    # Safety: numeric checks
    try:
        credit_limit = float(state.customer.get("credit_limit", 0))
        outstanding = float(state.customer.get("outstanding_balance", 0))
        order_amount = float(state.order.get("order_amount", 0))
        available = credit_limit - outstanding
        would_exceed = order_amount > max(available, 0)
    except Exception:
        available = None
        would_exceed = None

    # Guardrails
    if decision == "approved" and would_exceed is True:
        decision = "escalate"
        reason = (f"Safety check: order_amount exceeds available credit. "
                  f"Available={available}, Order={state.order.get('order_amount')}.")

    if decision not in ("approved", "rejected", "escalate"):
        decision = "escalate"
        reason = reason + " (Unrecognized decision; escalated.)"

    state.approval_status = decision  # type: ignore
    state.decision_reason = reason
    return state

def node_escalate_or_finish(state: ARState) -> ARState:
    if state.approval_status == "escalate":
        payload = {
            "order_id": state.order_id,
            "customer_id": state.customer_id,
            "order": state.order,
            "customer": state.customer,
            "policy_excerpt": state.policy_text[:1200],
            "llm_assessment": state.credit_assessment,
            "reason": state.decision_reason,
            "instructions": (
                "Human review required. Add `human_note` and set `approval_status` "
                "to 'approved' or 'rejected' to finalize."
            ),
        }
        logging.warning("[escalate] Human review required for order_id=%s", state.order_id)

        # Prefer real Interrupt when available
        try:
            from langgraph.types import Interrupt  # v0.2+
            raise Interrupt(value=payload)
        except Exception:
            # Fallback: write a review packet
            folder = "escalations"
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"escalate_{state.order_id or 'unknown'}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logging.warning("[escalate] Wrote human review packet: %s", path)
    else:
        logging.info("[finish] order_id=%s → %s", state.order_id, state.approval_status)
    return state

def node_log(state: ARState) -> ARState:
    logging.info("[log] order_id=%s status=%s reason=%s",
                 state.order_id, state.approval_status, state.decision_reason)
    return state

# --------------------------------------------------------------------------------------
# Graph builder
# --------------------------------------------------------------------------------------

def build_graph(llm: ChatOllama):
    g = StateGraph(ARState)
    g.add_node("prepare", node_prepare)
    g.add_node("llm_assess", lambda s: node_llm_assess(s, llm))
    g.add_node("decide", node_decide)
    g.add_node("escalate_or_finish", node_escalate_or_finish)
    g.add_node("log", node_log)

    g.set_entry_point("prepare")
    g.add_edge("prepare", "llm_assess")
    g.add_edge("llm_assess", "decide")
    g.add_edge("decide", "escalate_or_finish")
    g.add_edge("escalate_or_finish", "log")
    g.add_edge("log", END)

    memory = MemorySaver()
    return g.compile(checkpointer=memory)

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AR Agent (Ollama + Gemma 3 + LangGraph)")
    parser.add_argument("--customer_csv", default="customer_master.csv")
    parser.add_argument("--orders_csv", default="sales_order.csv")
    parser.add_argument("--policy", default="CreditPolicy.txt")
    parser.add_argument("--model", default="gemma3:latest")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    # Inputs
    policy_text = load_policy_text(args.policy)
    customer_map = load_customer_map(args.customer_csv)
    orders = load_orders(args.orders_csv)

    # LLM
    llm = make_llm(args.model, args.temperature)

    # Graph
    app = build_graph(llm)

    decisions: List[Dict[str, Any]] = []

    for order in orders:
        order_id = str(order.get("order_id", f"noid-{datetime.now().timestamp()}"))
        customer_id = str(order.get("customer_id", "")).strip()
        customer = customer_map.get(customer_id, {})

        init = ARState(customer=customer, order=order, policy_text=policy_text)
        config = {"configurable": {"thread_id": f"order-{order_id}"}}

        logging.info("=" * 80)
        logging.info("Processing order_id=%s, customer_id=%s", order_id, customer_id)

        try:
            raw_state = app.invoke(init, config=config)     # may return dict
            final_state = ensure_arstate(raw_state)         # normalize to ARState
        except Exception as e:
            # Interrupt or any error → escalate safely
            logging.warning("Interrupted / Exception for order %s: %s", order_id, e)
            final_state = init
            final_state.approval_status = "escalate"
            final_state.decision_reason = f"Interrupted for human review or error: {e}"

        decisions.append({
            "order_id": order_id,
            "customer_id": customer_id,
            "approval_status": final_state.approval_status,
            "decision_reason": final_state.decision_reason,
            "credit_assessment": json.dumps(final_state.credit_assessment or {}, ensure_ascii=False),
        })

    out_csv = write_decisions_csv(decisions, args.out_csv)
    logging.info("Wrote decisions CSV: %s", out_csv)
    logging.info("Workflow log: %s | LLM log: %s", WF_LOG_PATH, LLM_LOG_PATH)

if __name__ == "__main__":
    main()
