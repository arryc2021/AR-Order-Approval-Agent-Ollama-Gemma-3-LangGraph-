# 📊 AR Order Approval Agent (Ollama + Gemma 3 + LangGraph)

This project implements an **AI-powered Accounts Receivable (AR) workflow** that automates sales order approval decisions based on customer creditworthiness and company policy.

It uses:
- **[Ollama](https://ollama.ai/)** to run the **Gemma 3 LLM locally** (no cloud calls).
- **LangGraph** for a **stateful, traceable workflow** with human-in-the-loop support.
- **CSV + text files** as input (no database needed).

---

## 🎯 Objectives
- Automate AR checks for sales orders against customer credit limits and company policy.  
- Leverage **Gemma 3** for nuanced reasoning about credit risk.  
- Provide **traceability** with logs and decision outputs.  
- Support **human escalation** when orders cannot be auto-approved/rejected.  

---

## 📥 Inputs

### 1. `customer_master.csv`
```csv
customer_id,customer_name,credit_limit,outstanding_balance,payment_terms,segment,risk_rating
C001,Acme Corp,50000,12000,Net30,Enterprise,Medium
```

### 2. `sales_order.csv`
```csv
order_id,customer_id,order_amount,currency,order_date,requested_terms,sales_rep
SO-1001,C001,15000,USD,2025-08-25,Net30,alice@company.com
```

### 3. `CreditPolicy.txt`
```
- Orders must not exceed available credit (credit_limit - outstanding_balance).
- If risk_rating is High and order_amount > 10% of credit_limit, escalate.
- Enterprise segment may exceed by up to 5% with manager approval (escalate).
```

---

## 📤 Outputs

### 1. Decisions CSV
**`ar_decisions_<timestamp>.csv`**
```csv
order_id,customer_id,approval_status,decision_reason,credit_assessment
SO-1001,C001,approved,"Within credit limit",{"decision":"approved","computed":{"available_credit":38000}}
```

### 2. Logs
- **`ar_workflow_<timestamp>.log`** → workflow execution trace  
- **`llm_interactions.log`** → prompts + Gemma 3 responses  

### 3. Escalations
If Gemma 3 or numeric guardrails cannot finalize, a JSON packet is written:  
```
./escalations/escalate_SO-1001.json
```

A reviewer can open this file, add a `human_note` and finalize the decision.

---

## 🏗️ System Architecture

```
ar-order-approval-agent/
├── ar_agent.py           # main workflow (LangGraph + Ollama + Gemma 3)
├── customer_master.csv   # sample input (customers)
├── sales_order.csv       # sample input (orders)
├── CreditPolicy.txt      # company credit policy rules
├── requirements.txt      # Python dependencies
├── README.md             # GitHub documentation
└── escalations/          # (created during runtime) escalation packets
```

---

## 📦 Requirements

```
langchain>=0.2.0
langchain-community>=0.2.0
langgraph>=0.2.0
pydantic>=2.0
pandas>=2.0
python-dateutil>=2.8
```

---

## ⚙️ Setup

### 1. Install Ollama + Gemma 3
```bash
# Install Ollama (see https://ollama.ai/download for your OS)
ollama serve

# Pull the Gemma 3 model
ollama pull gemma3:latest
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Agent
```bash
python ar_agent.py   --customer_csv customer_master.csv   --orders_csv sales_order.csv   --policy CreditPolicy.txt   --model gemma3:latest
```

---

## 👩‍💼 Human-in-the-Loop

If an order cannot be auto-approved or rejected, it is escalated.  
Two modes:
1. **LangGraph Interrupt** → pauses workflow for reviewer input (if supported).
2. **Escalation Packet** → JSON file written under `./escalations/` for manual review.

Example packet:
```json
{
  "order_id": "SO-1002",
  "customer_id": "C001",
  "reason": "Risk rating High, requires manual approval",
  "llm_assessment": {...},
  "instructions": "Human needs to review this order."
}
```

Reviewer updates:
```json
"human_note": "Approved by manager due to long-standing customer relationship",
"approval_status": "approved"
```

---

## 📈 Performance Parameters

To evaluate and monitor the system:

| Metric                     | Description                                                                 | Typical Value* |
|-----------------------------|-----------------------------------------------------------------------------|----------------|
| **Throughput**              | Orders processed per minute (depends on hardware and LLM speed).            | ~5–10 orders/minute on CPU; higher on GPU |
| **Latency per order**       | Time to process a single order (CSV row) including LLM call.                 | 5–12 seconds (Gemma 3, CPU); 1–3 seconds (GPU) |
| **Escalation rate**         | % of orders requiring human review.                                          | 5–15% (depends on credit policy strictness) |
| **Accuracy of decisions**   | % of orders where AI decision matches human auditor.                         | 85–95% (validated against policy & historical data) |
| **Resource usage**          | CPU: moderate; GPU: recommended for large batches. Memory: ~4–6 GB.         | Varies |
| **Scalability**             | Horizontal scaling possible (split orders into batches across multiple agents). | Linearly scalable |
| **Auditability**            | 100% of prompts, responses, and final decisions logged locally.              | ✅ Always |

\*Values are indicative — adjust after benchmarking in your environment.

---

## 🚀 Future Enhancements
- Streamlit / FastAPI dashboard for reviewer workflow.  
- Policy versioning (log hash of CreditPolicy.txt with each decision).  
- Multi-currency handling with FX normalization.  
- Docker Compose for full containerized setup.  
- Prometheus/Grafana integration for performance monitoring.  

---

## 📜 License
MIT License © 2025
