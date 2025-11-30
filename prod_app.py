# app.py — AgentPM (Merged, deployment-ready)
import streamlit as st
import asyncio
import os
import json
import time
import logging
import inspect
import google.generativeai as genai
from typing import Dict, Any, List, Tuple
from difflib import SequenceMatcher
from duckduckgo_search import DDGS
import ast
import pandas as pd

import os
from dotenv import load_dotenv
load_dotenv()

# Prefer Streamlit secrets (when deployed on Streamlit Cloud)
def get_gemini_key():
    try:
        # st may not be imported yet in some contexts; guard
        import streamlit as _st
        val = _st.secrets.get("GEMINI_API_KEY")
        if val:
            return val
    except Exception:
        pass
    # fallback to env
    return os.getenv("GEMINI_API_KEY")

api_key = get_gemini_key()


# ------------------------
# Config & Working Dir
# ------------------------
st.set_page_config(page_title="AgentPM", page_icon="🚀", layout="wide")
WORKDIR = os.environ.get("AGENTPM_WORKDIR", "agentpm_state")
os.makedirs(WORKDIR, exist_ok=True)

# ------------------------
# Observability (from App2)
# ------------------------
if "traces" not in st.session_state: st.session_state.traces = []
if "evals" not in st.session_state: st.session_state.evals = []
if "logs" not in st.session_state: st.session_state.logs = []

def ui_log(role: str, message: str):
    st.session_state.logs.append({"role": role, "content": message})

class Trace:
    def __init__(self, name: str):
        self.name = name
        self.start = time.time()
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_value, traceback):
        duration = time.time() - self.start
        st.session_state.traces.append({
            "Agent": self.name,
            "Duration (s)": round(duration, 2),
            "Timestamp": time.strftime("%H:%M:%S"),
            "Status": "Error" if exc_type else "Success"
        })

# ------------------------
# Memory (from App1 — persistent MemoryBank + sessions)
# ------------------------
class InMemorySessionService:
    def __init__(self):
        self.sessions = {}
    def create(self, session_id: str):
        self.sessions[session_id] = {"created_at": time.time(), "history": []}
        return self.sessions[session_id]
    def get(self, session_id: str, default=None):
        return self.sessions.get(session_id, default)
    def update(self, session_id: str, key: str, value: Any):
        self.sessions.setdefault(session_id, {})[key] = value
    def append_history(self, session_id: str, role: str, content: str):
        if session_id in self.sessions:
            self.sessions[session_id].setdefault("history", []).append(f"{role}: {content}")

class MemoryBank:
    def __init__(self, path=os.path.join(WORKDIR, 'memory_bank.json')):
        self.path = path
        if os.path.exists(self.path):
            try:
                with open(self.path,'r') as f:
                    self.store = json.load(f)
            except Exception:
                self.store = {}
        else:
            self.store = {}
    def write(self, key: str, value: Any):
        self.store[key] = {'value': value, 'ts': time.time()}
        with open(self.path, 'w') as f:
            json.dump(self.store, f, indent=2)
    def read(self, key: str):
        return self.store.get(key)
    def query_similar(self, text: str, top_k: int = 3) -> List[Tuple[str,float,Dict]]:
        candidates = []
        for k,v in self.store.items():
            ratio = SequenceMatcher(None, text.lower(), str(v['value']).lower()).ratio()
            candidates.append((k, ratio, v))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

# init
if "memory" not in st.session_state:
    st.session_state.memory = MemoryBank()
if "sessions" not in st.session_state:
    st.session_state.sessions = InMemorySessionService()

# ------------------------
# ToolRegistry (from App1, async-friendly)
# ------------------------
class ToolRegistry:
    def __init__(self):
        self.tools = {}
    def register(self, func):
        schema = {"name": func.__name__, "doc": func.__doc__, "sig": str(inspect.signature(func))}
        self.tools[func.__name__] = {"func": func, "schema": schema}
        return func
    async def execute(self, tool_name: str, *args):
        if tool_name in self.tools:
            func = self.tools[tool_name]["func"]
            try:
                if inspect.iscoroutinefunction(func):
                    return await func(*args)
                return func(*args)
            except Exception as e:
                return f"Tool execution error: {e}"
        return f"Tool {tool_name} not found."

tool_registry = ToolRegistry()

# ------------------------
# Real DuckDuckGo Search Tool (from App2), cleaned output (English oriented)
# ------------------------
@tool_registry.register
def duckduckgo_search(query: str, max_results: int = 6) -> str:
    """
    REAL web search via DuckDuckGo (DDGS).
    Returns a cleaned, English-friendly multi-line summary suitable for LLM consumption.
    """
    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))
        # fallback: broaden query if empty
        if not results:
            # try a broader form (remove punctuation, split tokens)
            fallback_q = " ".join([t for t in query.replace("/", " ").split() if len(t) > 2])
            results = list(ddgs.text(fallback_q, max_results=max_results))
        if not results:
            return "No web results found for that query."
        # Format into clean English text
        items = []
        for r in results:
            title = (r.get("title") or "").strip()
            body = (r.get("body") or "").replace("\n", " ").strip()
            preview = (title or body[:120]).strip()
            items.append(f"- {preview}: {body[:240]}")
        return "REAL WEB RESULTS (English summary):\n" + "\n".join(items)
    except Exception as e:
        return f"Search failed: {e}"


# ------------------------
# Safe calculator (security: avoid eval) — use ast.literal_eval
# ------------------------
@tool_registry.register
def calculator(expression: str) -> str:
    """Safe calculator (supports numeric expressions)."""
    try:
        # ast.literal_eval only supports literals, so we try simple math via eval but with safety:
        # implement a tiny parser using ast (restrict to math ops)
        import ast, operator as op
        # supported operators
        operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
                     ast.Pow: op.pow, ast.USub: op.neg, ast.Mod: op.mod}
        def eval_expr(node):
            if isinstance(node, ast.Num): return node.n
            if isinstance(node, ast.BinOp):
                return operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
            if isinstance(node, ast.UnaryOp):
                return operators[type(node.op)](eval_expr(node.operand))
            raise ValueError("Unsupported expression")
        node = ast.parse(expression, mode='eval').body
        return str(eval_expr(node))
    except Exception as e:
        return f"Calculator error: {e}"

# ------------------------
# LLM Client (Gemini-ready from App1 with safe fallback)
# ------------------------
GEMINI_AVAILABLE = False
try:
    # configure if key present in env; callers may pass key also
    if os.environ.get("GEMINI_API_KEY"):
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

class LLMClient:
    def __init__(self, api_key: str = None, model_name: str = "models/gemini-2.5-flash"):
        self.model_name = model_name
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.available = True
        else:
            # fallback: use global env config if present
            self.available = GEMINI_AVAILABLE
            self.model = genai.GenerativeModel(model_name) if GEMINI_AVAILABLE else None

    async def generate(self, system_prompt: str, user_prompt: str, context: str) -> str:
        # Use a controlled protocol; agent can output TOOL_CALL: tool(arg)
        tool_instructions = """
PROTOCOL:
- If you need to use a tool, output exactly this line (no extra text):
  TOOL_CALL: tool_name('argument')
- Otherwise, output your final response in English only.
"""
        full_prompt = f"{tool_instructions}\nSYSTEM: {system_prompt}\nCONTEXT: {context}\nTASK: {user_prompt}"
        if self.available and self.model:
            try:
                # use async API if available
                resp = await self.model.generate_content_async(full_prompt)
                return resp.text
            except Exception as e:
                return f"[LLM error]: {e}"
        # fallback mock (keeps outputs English and concise)
        return f"[MOCK LLM reply to]: {user_prompt[:240]}"

# ------------------------
# Context compaction (from App1)
# ------------------------
async def context_compactor(session_id: str, max_chars: int = 2000) -> str:
    sess = st.session_state.sessions.get(session_id, {})
    history_str = "\n".join(sess.get("history", [])[-10:])
    relevant = []
    if history_str:
        sims = st.session_state.memory.query_similar(history_str, top_k=2)
        for k,score,entry in sims:
            relevant.append(f"Memory({k}): {entry['value']}")
    joined = "\n".join(relevant)
    compacted = (history_str + "\n\n" + joined)[:max_chars]
    return compacted or "No context."

# ------------------------
# Agents (based on App1)
# ------------------------
class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.orch = None
    async def handle(self, message: Dict[str,Any]) -> Dict[str,Any]:
        raise NotImplementedError
    async def send_to_memory(self, key: str, value: Any):
        st.session_state.memory.write(key, value)
        return True

class ResearchAgent(BaseAgent):
    def __init__(self, name, tool_registry, llm: LLMClient):
        super().__init__(name)
        self.tools = tool_registry
        self.llm = llm
    async def handle(self, message: Dict[str,Any]):
        with Trace(self.name):
            query = message.get("content","research")
            session_id = message.get("session_id","default")
            # call web search tool
            tool_res = await self.tools.execute("duckduckgo_search", query)
            # normalize to English sections similar to App1 mock
            report = f"Research results for: {query}\n\n{tool_res}"
            await self.send_to_memory(f"research:{query}", report)
            st.session_state.sessions.append_history(session_id, self.name, report)
            ui_log(self.name, report)
            return {"content": report, "session_id": session_id}

class CompetitorAgent(BaseAgent):
    def __init__(self, name, tool_registry):
        super().__init__(name)
        self.tools = tool_registry
    async def handle(self, message: Dict[str,Any]):
        with Trace(self.name):
            domain = message.get("content","competitor.com")
            res = await self.tools.execute("duckduckgo_search", domain)
            summary = f"Competitor summary for {domain}:\n{res}"
            await self.send_to_memory(f"competitor:{domain}", summary)
            session_id = message.get("session_id","default")
            st.session_state.sessions.append_history(session_id, self.name, summary)
            ui_log(self.name, summary)
            return {"content": summary, "session_id": session_id}

class PlannerAgent(BaseAgent):
    def __init__(self, name, llm: LLMClient):
        super().__init__(name)
        self.llm = llm
    async def handle(self, message: Dict[str,Any]):
        with Trace(self.name):
            session_id = message.get("session_id","default")
            ctx = await context_compactor(session_id)
            prompt = f"Create a concise product plan (goals, milestones, risks) from:\n{message.get('content')}"
            plan = await self.llm.generate("You are a product planner.", prompt, ctx)
            await self.send_to_memory("latest_plan", plan)
            st.session_state.sessions.append_history(session_id, self.name, plan)
            ui_log(self.name, plan)
            return {"content": plan, "session_id": session_id}

class SpecWriterAgent(BaseAgent):
    def __init__(self, name, llm: LLMClient, tool_registry: ToolRegistry):
        super().__init__(name)
        self.llm = llm
        self.tools = tool_registry
    async def handle(self, message: Dict[str,Any]):
        with Trace(self.name):
            session_id = message.get("session_id","default")
            ctx = await context_compactor(session_id)
            prompt = f"Write a developer-facing spec based on:\n{message.get('content')}"
            spec = await self.llm.generate("You are a CTO. Write specs in clear English.", prompt, ctx)
            # create multi-section final spec similar to App1
            final_spec = f"TECHNICAL SPECIFICATION\n\n{spec}\n\n(End of Spec)"
            await self.send_to_memory("latest_spec", final_spec)
            st.session_state.sessions.append_history(session_id, self.name, final_spec)
            ui_log(self.name, final_spec)
            return {"content": final_spec, "session_id": session_id}

class MemoryAgent(BaseAgent):
    async def handle(self, message: Dict[str,Any]):
        t = message.get("type")
        if t == "memory_write":
            st.session_state.memory.write(message["key"], message["value"])
            return {"content":"ok"}
        elif t == "memory_read":
            v = st.session_state.memory.read(message["key"])
            return {"content": v}
        return {"content":"unknown"}

# ------------------------
# Orchestrator (from App2)
# ------------------------
class Orchestrator:
    def __init__(self):
        self.agents = {}
    def register_agent(self, agent: BaseAgent):
        self.agents[agent.name] = agent
    async def run_parallel(self, names: List[str], message: Dict):
        tasks = [self.agents[n].handle(message) for n in names]
        return await asyncio.gather(*tasks)
    async def run_sequential(self, names: List[str], initial_message: Dict):
        msg = initial_message
        for n in names:
            res = await self.agents[n].handle(msg)
            msg = res
        return msg
    async def run_loop(self, name: str, message: Dict, max_iters: int = 3, session_id: str | None = None):
        """
        Run the named agent in a loop up to max_iters.
        If session_id is provided, pass it into agent.handle as 'session_id'.
        """
        msg = message
        for i in range(max_iters):
            # build call args: if agent.handle expects session_id in message dict, include it
            if session_id is not None:
                # ensure msg is a dict and include session_id
                if isinstance(msg, dict):
                    msg['session_id'] = session_id
                else:
                    msg = {"content": msg, "session_id": session_id}
            res = await self.agents[name].handle(msg)
            # normalize res to dict with 'content'
            if not isinstance(res, dict):
                res = {"content": res}
            if "DONE" in res.get("content", ""):
                return res
            # prepare next iteration message
            msg = {"content": res["content"], "session_id": session_id}
        return msg


# ------------------------
# Long-running op example (from App1) — pause/resume
# ------------------------
async def long_running_research(task_id: str, query: str, total_steps: int = 5):
    state_file = os.path.join(WORKDIR, f'lr_{task_id}.json')
    if os.path.exists(state_file):
        with open(state_file,'r') as f:
            state = json.load(f)
        current = state.get('current', 0)
    else:
        current = 0
    for step in range(current, total_steps):
        # simulate work chunk
        time.sleep(0.2)
        with open(state_file,'w') as f:
            json.dump({'current': step+1}, f)
    # write final result
    st.session_state.memory.write(f'long_research:{task_id}', {'query': query, 'result': 'final results'})

# ------------------------
# Evaluation utilities (both apps)
# ------------------------
def similarity_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def compile_full_report(project_name: str, research: str, spec: str, qa: str, estimate: str) -> str:
    return f"""
==================================================
        AGENT PM - PROJECT REPORT
==================================================

Project: {project_name}

[1] RESEARCH
{research}

[2] SPECIFICATION
{spec}

[3] QA
{qa}

[4] ESTIMATION
{estimate}

==================================================
Generated by AgentPM
"""

# ------------------------
# UI: combine chat-style logs + pages (Workflow + Dashboard)
# ------------------------
with st.sidebar:
    st.header("AgentPM Config")
    api_key = st.text_input("Gemini API Key", type="password")
    st.markdown("**Note:** Provide GEMINI_API_KEY via env for deployment.")
    st.divider()
    page = st.radio("Page", ["Workflow","Dashboard"])
    if st.button("Reset All"):
        for k in ["logs","traces","evals"]:
            st.session_state[k] = []
        st.session_state.sessions = {}
        st.session_state.memory = MemoryBank()
        st.rerun()

# Chat-style output area
for log in st.session_state.logs:
    role = log["role"]
    content = log["content"]
    if role == "system":
        st.caption(content)
    else:
        with st.chat_message(role):
            st.markdown(content)

# Workflow page
if page == "Workflow":
    st.title("AgentPM — Workflow")
    if not api_key and not GEMINI_AVAILABLE:
        st.warning("Enter Gemini API Key or set GEMINI_API_KEY in environment before using real LLM. Running in mock mode.")
    col1, col2 = st.columns(2)
    with col1:
        p_name = st.text_input("Product Name", "DevTok")
        p_audience = st.text_input("Target Audience", "Developers")
        p_features = st.text_area("Key Features", "60s tutorials, IDE plugin")
        p_stack = st.text_input("Tech Stack", "Flutter & Go")
    if "step" not in st.session_state: st.session_state.step = 0
    if "spec" not in st.session_state: st.session_state.spec = ""
    if "research" not in st.session_state: st.session_state.research = ""
    if st.session_state.step == 0:
        if st.button("Launch Agents (Research -> Plan -> Spec)"):
            llm = LLMClient(api_key)  # if api_key blank, falls back to mock
            tools = tool_registry
            orch = Orchestrator()
            # Register agents
            r = ResearchAgent("Researcher", tools, llm)
            c = CompetitorAgent("Competitor", tools)
            p = PlannerAgent("Planner", llm)
            s = SpecWriterAgent("SpecWriter", llm, tools)
            m = MemoryAgent("Memory")
            for agent in [r,c,p,s,m]:
                orch.register_agent(agent)
            sid = "sess_main"
            st.session_state.sessions.create(sid)
            async def run_phase():
                ui_log("system","--- Phase 1: Parallel Research ---")
                res = await orch.run_parallel(["Researcher","Competitor"], {"content": f"{p_name} competitors & stack", "session_id": sid})
                combined = "\n\n".join([r['content'] for r in res])
                st.session_state.research = combined
                ui_log("system","--- Phase 2: Planner & SpecWriter (Sequential) ---")
                plan = await orch.run_sequential(["Planner"], {"content": combined, "session_id": sid})
                spec = await orch.run_sequential(["SpecWriter"], {"content": plan['content'], "session_id": sid})
                st.session_state.spec = spec['content']
            asyncio.run(run_phase())
            st.session_state.step = 1
            st.rerun()
    elif st.session_state.step == 1:
        st.subheader("Human Review: Edit Spec")
        edited = st.text_area("Edit generated spec:", value=st.session_state.spec, height=300)
        c1, c2 = st.columns(2)
        if c1.button("Approve Spec"):
            st.session_state.spec = edited
            st.session_state.step = 2
            st.rerun()
        if c2.button("Request Revision"):
            llm = LLMClient(api_key)
            spec_agent = SpecWriterAgent("SpecWriter", llm, tool_registry)
            async def revise_spec():
                res = await spec_agent.handle({"content": edited, "session_id":"sess_main"})
                st.session_state.spec = res['content']
            asyncio.run(revise_spec())
            st.rerun()
    elif st.session_state.step == 2:
        st.subheader("QA & Estimation")
        if st.button("Run QA & Estimation"):
            llm = LLMClient(api_key)
            orch = Orchestrator()
            qa = PlannerAgent("QA", llm)  # reuse Planner style for QA prompting
            est = PlannerAgent("Estimator", llm)
            for a in [qa, est]:
                orch.register_agent(a)
            async def run_qae():
                ui_log("system","--- QA Loop ---")
                qa_res = await orch.run_loop("QA", {"content": st.session_state.spec, "session_id":"sess_main"}, session_id="sess_main")
                ui_log("system","--- Estimation ---")
                est_res = await est.handle({"content": qa_res['content'], "session_id":"sess_main"})
                # compile report
                report = compile_full_report(p_name, st.session_state.research, st.session_state.spec, qa_res['content'], est_res['content'])
                st.session_state.final_report = report
                ui_log("system","Project complete — report ready.")
            asyncio.run(run_qae())
            st.session_state.step = 3
            st.rerun()
    elif st.session_state.step == 3:
        st.success("Workflow Complete!")
        st.download_button("Download Full Report", st.session_state.final_report or "No report", file_name=f"{p_name}_AgentPM_Report.txt")
        rating = st.slider("Rate this session", 1, 5, 4)
        if st.button("Submit Rating"):
            st.session_state.evals.append({"Project": p_name, "Rating": rating, "Time": time.strftime("%H:%M")})
            st.success("Saved")

# Dashboard page
elif page == "Dashboard":
    st.title("Observability Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Traces", len(st.session_state.traces))
    avg_rating = (sum([e['Rating'] for e in st.session_state.evals]) / len(st.session_state.evals)) if st.session_state.evals else 0
    col2.metric("Avg Rating", f"{avg_rating:.1f}/5")
    col3.metric("Sessions", len(st.session_state.sessions.sessions))
    st.subheader("Traces")
    if st.session_state.traces:
        df = pd.DataFrame(st.session_state.traces)
        st.bar_chart(df.set_index("Agent")["Duration (s)"])
        with st.expander("Raw Traces"):
            st.dataframe(df)
    else:
        st.info("No traces yet.")
    st.subheader("Ratings")
    if st.session_state.evals:
        st.dataframe(pd.DataFrame(st.session_state.evals))
    else:
        st.info("No ratings yet.")

# ------------------------
# End of app.py
# ------------------------
