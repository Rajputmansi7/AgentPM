<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/2d971d67-c68b-43a2-9847-f928a01750b2" />
#  AgentPM — Multi-Agent AI Product Manager

AgentPM is a **production-ready multi-agent system** that automates the entire product management workflow — from research to planning, specs, QA, and estimation.

Built using **Python, Streamlit, LLMs (Gemini), DDGS Web Search, async orchestration, and custom tool/memory systems**, AgentPM helps small teams or solo developers generate PM-quality product documents in minutes.

---

##  **Features**

✔ **Real-time market + competitor research**
✔ **Autonomous multi-agent collaboration**
✔ **Parallel, sequential, and looped execution flows**
✔ **Developer-ready technical specifications**
✔ **QA review loops (iterative self-correction)**
✔ **Effort/timeline estimation**
✔ **Persistent memory bank (JSON-based)**
✔ **Full observability dashboard (logs, traces, ratings)**
✔ **Downloadable end-to-end PM report**
✔ **Production-ready Streamlit deployment**

---

##  **Architecture Overview**

AgentPM uses 6 core agents + a memory agent:

### **Phase 1 — Parallel Execution**

* **ResearchAgent** → Web search, market insights
* **CompetitorAgent** → Competitor analysis

### **Phase 2 — Sequential Execution**

* **PlannerAgent** → Goals, milestones, risks
* **SpecWriterAgent** → Full tech spec

### **Phase 3 — QA Loop + Estimation**

* **QA Agent** → Iterative improvements until DONE
* **EstimatorAgent** → Timeline, effort, team size
* **MemoryAgent** → Persistent storage for all phases

---

##  **Multi-Agent Flow (ASCII Diagram)**

```
                ┌─────────────────┐        Phase 1 (Parallel)
                │ ResearchAgent   │
                └───────┬─────────┘
                        │
                        │
                ┌───────▼─────────┐
                │ CompetitorAgent  │
                └───────┬─────────┘
                        │ Combined Output
                        ▼
                ┌───────────────────────┐    Phase 2 (Sequential)
                │     PlannerAgent      │
                └───────────┬──────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │    SpecWriterAgent    │
                └───────────┬──────────┘
                            │
                            ▼
                ┌───────────────────────┐   Phase 3 (QA Loop)
                │        QA Agent       │◄───┐
                └───────────┬──────────┘    │ Loop until approved
                            │                │
                            ▼                │
                ┌───────────────────────┐    │
                │   EstimatorAgent      │─────┘ Sequential
                └───────────┬──────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │     MemoryAgent       │
                └───────────────────────┘
```

---

##  **Project Structure**

```
agentpm/
│
├── prod_app.py               # Main Streamlit app (workflow + dashboard)
├── agentpm_state/            # Persistent memory & long-running state
├── requirements.txt
├── README.md
└── LICENSE
```

---

##  **Tech Stack**

| Component     | Technology                                    |
| ------------- | --------------------------------------------- |
| UI            | Streamlit                                     |
| LLM           | Gemini 2.5 Flash                              |
| Search        | DDGS (DuckDuckGo Search)                      |
| Agents        | Python async coroutines                       |
| Memory        | Custom JSON-backed MemoryBank                 |
| Orchestration | Parallel, sequential & loop-based agent flows |
| Observability | Logs, traces, dashboard                       |
| Deployment    | Streamlit Cloud / Local                       |

---

##  **Installation**

### **1. Clone the repository**

```bash
git clone https://github.com/Rajputmansi7/AgentPM
cd agentpm
```

### **2. Create a virtual environment**

```bash
python -m venv agent
source agentbin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### **3. Install dependencies**

```bash
pip install -r requirements.txt
```

### **4. Add your Gemini API Key**

Create `.env`:

```
GEMINI_API_KEY=your_key_here
```

Or set it directly in Streamlit Cloud.

### **5. Run the app**

```bash
streamlit run prod_app.py
```

---

##  **Usage**

### **Step 1 — Enter Product Details**

* Product Name
* Target Audience
* Key Features
* Tech Stack

### **Step 2 — Run Agents**

Click **Launch Agents** → this triggers:

1. Research + Competitor agents (parallel)
2. Planning + Spec writing (sequential)
3. QA loop
4. Estimation

### **Step 3 — Edit & Approve Spec**

Human-in-the-loop editing allowed.

### **Step 4 — Download Final Report**

Includes:

* Research
* Specification
* QA review
* Estimation

### **Step 5 — View Observability Dashboard**

* Agent traces
* Ratings
* Logs

---

##  **Custom Tools Included**

### **🔍 duckduckgo_search(query)**

Real web search using DDGS.

### ** calculator(expression)**

Safe arithmetic evaluator using AST parsing.

*Note:* These tools are part of a **custom ToolRegistry**, not MCP.

---

##  **Memory System**

AgentPM includes:

* Persistent JSON-based memory
* Similarity search for context compaction
* Automatic context loading per agent

This enables long-term cross-agent coherence.

---

##  **Long-Running Tasks**

A prototype long-running agent is included:

```python
long_running_research(task_id, query)
```

This demonstrates:

* Checkpointing
* Resume-on-refresh
* Background work simulation

---

##  **Error Handling & Production Hardening**

AgentPM includes:

* Centralized logging
* Trace analytics
* DDGS fallback search
* Graceful LLM fallback mode
* Session resets
* Async task orchestration fixes
* gRPC runtime stability mitigation

This ensures reliable production behavior.

---

##  **Deployment**

AgentPM supports:

* **Streamlit Cloud** (recommended)
* Any cloud VM (AWS, GCP, Azure)
* Local hosting

Set environment variables in deployment:

```
GEMINI_API_KEY= access from Google AI Studio
```

---

##  **Final Project Report**

After running all agents, AgentPM automatically generates:

* Research Summary
* Competitor Summary
* Product Plan
* Technical Specification
* QA Pass Results
* Engineering Estimation
* Complete Project Document

Downloadable as a full text report.

---

##  **Contributing**

Pull requests are welcome!

If you want to:

* Add new agents
* Add new tools
* Improve UI
* Improve LLM prompts

Feel free to contribute.

---

Deployed on Streamlit Cloud: https://agentpm.streamlit.app/
