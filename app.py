import asyncio
import json
import os
from contextlib import asynccontextmanager
import time
from pathlib import Path
import logging
import sys
import traceback
import re
import streamlit as st
from openai import AzureOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ------------------ test langgraph ------------------

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Optional, List, Dict, Any



MAX_RETRIES = 1

class AgentState(TypedDict, total=False):
    user_msg: str               # ← set once at entry only
    plan: Dict[str, Any]
    sql: Optional[str]
    tables: List[str]
    viz_code: Optional[str]
    df_json: Optional[str]
    answer: Optional[str]
    error: Optional[str]
    retry_count: int

def build_langgraph(client, deployment, session):
    g = StateGraph(AgentState)

    async def planner_node(state: AgentState):
        user_msg = state["user_msg"]  # READ ONLY
        sys = (
            SYS_PROMPT
            + "\nReturn ONLY a compact JSON object with keys: "
              "need_db (bool), sql (string or null), tables (array of strings), "
              "viz_code (string or null). No extra text."
        )
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        plan: Dict[str, Any] = {}
        try:
            if raw.startswith("```"):
                raw = raw.strip("`")
                raw = raw.split("\n", 1)[-1]
                if raw.startswith("json"):
                    raw = raw[4:]
            if "{" in raw and "}" in raw:
                raw = raw[raw.index("{"): raw.rindex("}")+1]
            plan = json.loads(raw)
        except Exception:
            plan = {}

        return {
            "plan": plan,
            "sql": plan.get("sql"),
            "viz_code": plan.get("viz_code"),
            "tables": plan.get("tables") or [],
            "retry_count": state.get("retry_count", 0),
        }

    g.add_node("planner", planner_node)

    def need_db_router(state: AgentState):
        return "introspect" if bool(state.get("plan", {}).get("need_db", True)) else "summarize"
    g.add_conditional_edges("planner", need_db_router, {"introspect": "introspect", "summarize": "summarize"})

    async def introspect_node(state: AgentState):
        tables = state.get("tables") or []

        # light table guess from SQL if none provided
        sql_lower = (state.get("sql") or "").lower()
        if not tables and " from " in sql_lower:
            try:
                after_from = sql_lower.split(" from ", 1)[1]
                first = after_from.split()[0].strip(",;")
                if first:
                    tables = [first.replace('"', '').replace("'", "")]
            except Exception:
                pass

        missing: List[str] = []
        for t in tables:
            schema_table = t if "." in t else f"public.{t}"
            try:
                res = await session.call_tool("query", {
                    "sql": f"SELECT to_regclass('{schema_table}') IS NOT NULL AS exists;"
                })
                exists = "true" in str(res.content).lower() or "t" in str(res.content).lower()
                if not exists:
                    missing.append(t)
            except Exception as e:
                return {"error": f"Introspection error for {t}: {e}"}

        plan = dict(state.get("plan", {}))
        plan["_introspect_missing"] = missing
        delta: AgentState = {"plan": plan}
        if missing:
            delta["error"] = f"Missing tables: {', '.join(missing)}"
        return delta

    g.add_node("introspect", introspect_node)

    def introspect_router(state: AgentState):
        return "error" if state.get("error") else "query"
    g.add_conditional_edges("introspect", introspect_router, {"error": "error", "query": "query"})

    async def query_node(state: AgentState):
        sql = state.get("sql")
        if not sql:
            return {"error": "Planner did not provide SQL."}

        # add LIMIT if missing
        if " limit " not in sql.lower():
            sql = sql.rstrip(" ;") + " LIMIT 100;"

        try:
            res = await session.call_tool("query", {"sql": sql})
            preview = str(res.content)
            return {"sql": sql, "df_json": preview}
        except Exception as e:
            return {"sql": sql, "error": f"Query failed: {e}"}

    g.add_node("query", query_node)

    def query_router(state: AgentState):
        if state.get("error"):
            return "error"
        return "visualize" if state.get("viz_code") else "summarize"
    g.add_conditional_edges("query", query_router, {"error": "error", "visualize": "visualize", "summarize": "summarize"})

    async def visualize_node(state: AgentState):
        code = state.get("viz_code")
        if not code:
            return {}
        try:
            handle_st_exec({"code": code, "dataframe_json": state.get("df_json")})
            return {"answer": "Rendered a visualization from the latest query."}
        except Exception as e:
            return {"error": f"Visualization error: {e}"}

    g.add_node("visualize", visualize_node)

    def viz_router(state: AgentState):
        return "error" if state.get("error") else "summarize"
    g.add_conditional_edges("visualize", viz_router, {"error": "error", "summarize": "summarize"})

    async def summarize_node(state: AgentState):
        if state.get("error"):
            return {"answer": state["error"]}

        # brief wrap-up
        msgs = [
            {"role": "system", "content": "Be concise and factual."},
            {"role": "user", "content": (
                "Provide a short summary of the result or the visualization that was produced. "
                "Do not include raw SQL or JSON."
            )},
        ]
        resp = client.chat.completions.create(model=deployment, messages=msgs, temperature=0.0)
        return {"answer": resp.choices[0].message.content}

    g.add_node("summarize", summarize_node)
    g.add_edge("summarize", END)

    # ── ERROR NODE ─────────────────────────────
    async def error_node(state: AgentState):
        # bump retry counter here
        retry = int(state.get("retry_count", 0)) + 1
        # Optional: clear the error so we don’t loop forever on same message
        return {"retry_count": retry, "error": state.get("error")}  # keep error text for summarize if needed

    g.add_node("error", error_node)

    def error_router(state: AgentState):
        retry = int(state.get("retry_count", 0))
        if retry <= MAX_RETRIES:
            return "planner"     # re-plan after error
        return "summarize"       # give up → explain error
    g.add_conditional_edges("error", error_router, {"planner": "planner", "summarize": "summarize"})

    g.set_entry_point("planner")
    return g.compile()



async def chat_turn_langgraph():
    user_msg = st.session_state.messages[-1]["content"]
    async with mcp_session() as session:
        graph = build_langgraph(client, deployment, session)
        state = {"user_msg": user_msg}
        result = await graph.ainvoke(state)
        answer = result.get("answer", "")
        return answer


# ------------------ test langgraph end ------------------


from toolbox import ST_EXEC_TOOL, handle_st_exec
# ----------------------------
# Streamlit page setup
# ----------------------------
st.set_page_config(page_title="MCP + Azure OpenAI", page_icon="🛠️", layout="wide")
st.title("🛠️ MCP + Azure OpenAI (Postgres tools)")

# ----------------------------
# Sidebar: config
# ----------------------------
with st.sidebar:
    st.header("Configuration")
    st.caption("Put secrets in `.streamlit/secrets.toml` – see example below.")

    # Azure config (prefer st.secrets)
    endpoint = st.text_input(
        "Azure OpenAI Endpoint",
        value=st.secrets.get("azure", {}).get("endpoint", "https://example.openai.azure.com/"),
    )
    api_version = st.text_input(
        "API Version",
        value=st.secrets.get("azure", {}).get("api_version", "2024-12-01-preview"),
    )
    azure_api_key = st.secrets.get("azure", {}).get("api_key") or os.getenv("AZURE_OPENAI_KEY")
    if not azure_api_key:
        st.warning("No Azure API key found. Add to `st.secrets['azure']['api_key']`.", icon="⚠️")

    e2b_api_key = st.secrets.get("e2b", {}).get("e2b_api_key") or os.getenv("E2B_API_KEY")
    if not e2b_api_key:
        st.warning("No E2B API key found. Add to `st.secrets['e2b']['e2b_api_key']`.", icon="⚠️")
    else:    
        st.caption('Received nonempty e2b api key.')
    # Deployment name to use in chat.completions.create(model=...)
    deployment = st.text_input(
        "Deployment (model) name",
        value=st.secrets.get("azure", {}).get("deployment", "gpt-5-chat"),
    )

    st.divider()
    st.header("About")
    st.markdown('TODO')
# =========================
# Cached Azure client
# =========================
@st.cache_resource(show_spinner=False)
def get_client(endpoint_: str, api_key_: str, api_version_: str) -> AzureOpenAI:
    return AzureOpenAI(api_version=api_version_, azure_endpoint=endpoint_, api_key=api_key_)

client = get_client(endpoint, azure_api_key or "MISSING_KEY", api_version)


# =========================
# Session state & logging
# =========================
SYS_PROMPT = (
""" 
You are an expert data analyst assisting a user with their Postgres database. You have access to tools to run SQL queries and visualize data.
TOOL-FIRST POLICY
- If the answer depends on database content, you MUST call `query` before replying. Do not answer from memory about data values, row counts, or schema.
- Never print raw SQL or JSON in your assistant message. Use tools. Summarize results in natural language only.
- Do not claim to have used a tool unless a tool call exists in this turn.

DECISION RECORD
Before answering, output exactly one line labeled DECISION containing a compact JSON object:
DECISION {"need_db": true|false, "goal": "<very short>", "sql": "<planned or empty>", "viz": true|false}
Keep it on one line. Then proceed with tool calls as needed.

SCHEMA DISCOVERY & EXISTENCE CHECKS
- When a table/column is not explicitly given or uncertain:
  1) List available tables:
     SELECT schemaname, tablename
     FROM pg_catalog.pg_tables
     WHERE schemaname NOT IN ('pg_catalog','information_schema')
     ORDER BY 1,2;
  2) Verify existence before use:
     SELECT to_regclass('public.<table>') IS NOT NULL AS exists;
  3) Inspect columns:
     SELECT column_name, data_type
     FROM information_schema.columns
     WHERE table_name='<table>'
     ORDER BY ordinal_position;

QUERY HYGIENE
- Add LIMIT 100 by default. Ask the user before large/expensive queries.
- Prefer selective predicates. If a query would scan very large tables without filters, ask for constraints.
- If no rows are returned, say so and suggest adjustments rather than guessing new tables.

VISUALIZATION
- If a chart helps, set "viz": true in DECISION, then call st_exec with concise code using st.line_chart / st.bar_chart / st.scatter_chart or matplotlib/plotly.
- Assume df exists from the last query; otherwise pass dataframe_json.

ERRORS & RETRIES
- On SQL error: fix and retry once. If it still fails, report the error briefly and ask how to proceed.
- On permission or connectivity errors: report and stop.
- On ambiguous requests: ask a clarifying question before querying.

TRUTHFULNESS
- If the required table or column does not exist (per to_regclass/INFORMATION_SCHEMA), say so plainly.

EXAMPLES
User: “Show the top 10 customers by revenue.”
Assistant:
DECISION {"need_db": true, "goal": "Find top customers by revenue", "sql": "", "viz": true}
[query] SELECT to_regclass('public.orders') IS NOT NULL AS exists;
[query] SELECT customer_id, SUM(total_amount) AS revenue
        FROM orders
        GROUP BY customer_id
        ORDER BY revenue DESC
        LIMIT 10;
[st_exec]  # code that bar_charts df[['customer_id','revenue']]
Assistant: Here are the top 10 customers by revenue. Want to drill into a specific customer?

User: “Get rows from table foo.”
Assistant:
DECISION {"need_db": true, "goal": "Verify table then sample", "sql": "", "viz": false}
[query] SELECT to_regclass('public.foo') IS NOT NULL AS exists;
Assistant: The table `foo` does not exist. Here are available tables: …
"""
)
 
def init_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": SYS_PROMPT}]
    if "logs" not in st.session_state:
        st.session_state.logs = []

def log(line: str):
    st.session_state.logs.append(line)

def token_trim(messages, max_chars: int = 14000):
    """Simple, dependency-free history trimmer by character length."""
    sys = [m for m in messages if m["role"] == "system"][:1]
    rest = [m for m in messages if m["role"] != "system"]
    total = 0
    kept = []
    for m in reversed(rest):
        total += len(m.get("content", ""))
        if total > max_chars:
            break
        kept.append(m)
    return sys + list(reversed(kept))

def clean_model_output(text: str) -> str:
    # remove any line beginning with "DECISION"
    return re.sub(r"^DECISION\s*\{.*\}\s*$", "", text, flags=re.MULTILINE).strip()

BASE_DIR = Path(__file__).parent.resolve()
MCP_DIR = BASE_DIR / "mcp-servers" / "mcp-postgres"
INDEX_JS = MCP_DIR / "dist" / "index.js"

# Get database URI from Streamlit secrets or environment
db_uri = (
    st.secrets.get("db", {}).get("uri") 
    or os.getenv("POSTGRES_URI") 
    or "postgresql://testuser:testpass@postgres:5432/testdb" 
)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def mcp_session():
    """Create an MCP session with the PostgreSQL server."""
    # 1) Validate the binary exists
    if not INDEX_JS.is_file():
        st.error(f"MCP entrypoint not found at: {INDEX_JS}")
        raise FileNotFoundError(str(INDEX_JS))

    # 2) Build server parameters
    server_params = StdioServerParameters(
        command="node",
        args=[str(INDEX_JS), db_uri],  # Remove duplicate db_uri
        env={
            **os.environ,
            "NODE_ENV": "production",
        }
    )

    try:
        logger.info("Opening stdio_client...")
        async with stdio_client(server_params) as (read, write):
            logger.info("stdio_client opened, creating ClientSession...")
            async with ClientSession(read, write) as session:
                logger.info("Initializing session...")
                result = await session.initialize()
                logger.info(f"Session initialized: {result}")
                
                # List available tools
                tools_result = await session.list_tools()
                logger.info(f"Available tools: {[t.name for t in tools_result.tools]}")
                st.sidebar.success(f"✅ Connected! {len(tools_result.tools)} tools available")
                
                yield session
                
    except ExceptionGroup as eg:
        # Extract the actual errors from the ExceptionGroup
        logger.error("ExceptionGroup caught:")
        for i, exc in enumerate(eg.exceptions):
            logger.error(f"  Exception {i}: {type(exc).__name__}: {exc}")
            logger.error(f"  Traceback: {''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}")
        st.error(f"MCP Connection failed with {len(eg.exceptions)} error(s). Check logs for details.")
        raise
    except Exception as e:
        logger.error(f"Failed to connect to MCP server: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Failed to connect to MCP server: {type(e).__name__}: {e}")
        raise

# =========================
# Tools bridge
# =========================

async def get_openai_tools(session: ClientSession):
    tools_result = await session.list_tools()
    # Test a tool call directly
    logger.info(f"Tools result: {tools_result}")
    result = await session.call_tool("query", {"sql": "SELECT 1"})
    logger.info(f"Test query result: {result}")
    tools = tools_result.tools
    log(f"🧩 Raw tool schemas: {[getattr(t, 'inputSchema', {}) for t in tools]}")

    names = [t.name for t in tools]
    log(f"🧰 Discovered tools: {names}")

    openai_tools = []
    for t in tools:
        # Ensure we pass a well-formed JSON Schema for function parameters.
        schema = getattr(t, "inputSchema", None) or {"type": "object", "properties": {}}
        # Normalize to object schema
        if not isinstance(schema, dict) or schema.get("type") != "object":
            schema = {"type": "object", "properties": {}}

        # If the tool accepts an `sql` property, mark it required so the model knows to supply it.
        props = schema.get("properties", {})
        if "sql" in props and "required" not in schema:
            schema["required"] = ["sql"]

        openai_tools.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": schema,
            },
        })

    # keep your custom exec tool
    openai_tools.append(ST_EXEC_TOOL)

    log(f"🧰 Tools available: {[t['function']['name'] for t in openai_tools]}")
    # also log full function schemas to help debugging
    logger.debug(f"OpenAI function defs: {openai_tools}")
    return openai_tools


# =========================
# One chat turn with tools
# =========================
async def chat_turn():
    st.session_state.logs.clear()

    # Only system/user/assistant in the long-term memory
    runtime_messages = [
        m for m in st.session_state.messages
        if m["role"] in ("system", "user", "assistant", "tool")
    ]

    async with mcp_session() as session:
        openai_tools = await get_openai_tools(session)
        tool_choice = "auto" if openai_tools else "none"

        # First call
        response = client.chat.completions.create(
            model=deployment,
            messages=runtime_messages,
            tools=openai_tools,
            tool_choice=tool_choice,
        )

        # Tool loop
        while getattr(response.choices[0].message, "tool_calls", None):
            tool_msg = response.choices[0].message
            log(f"Received tool_calls: {tool_msg.tool_calls!r}")
            
            # 1) append assistant message WITH tool_calls
            assistant_with_calls = {
                "role": "assistant",
                "content": "",   # coerce None
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    } for tc in tool_msg.tool_calls
                ],
            }
            runtime_messages.append(assistant_with_calls)

            # 2) execute tools and append tool messages right after
            for tc in tool_msg.tool_calls:
                fname = tc.function.name
                fargs = json.loads(tc.function.arguments or "{}")
                log(f"🔧 {fname} | 📝 {fargs}")

                # normalize common aliases the model might emit
                alias = {
                    "functions.query": "query",
                    "functions.st_exec": "st_exec",
                    "query" : "query",
                    "st_exec": "st_exec",
                    # add anything else the model might invent
                }
                fname_norm = alias.get(fname, fname)

                log(f"🔧 {fname} → {fname_norm} | 📝 {fargs}")

                if fname_norm == "st_exec":
                    try:
                        result = handle_st_exec(fargs)              # renders directly in Streamlit
                        runtime_messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result),
                        })
                    except Exception as e:
                        runtime_messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": f"Error in st_exec: {type(e).__name__}: {e}",
                        })
                        runtime_messages.append({
                            "role": "assistant",    
                            "content": f"Error during st_exec execution: {type(e).__name__}: {e}",
                        })
                    continue
                else:
                    try:
                        fname_norm = fname_norm.strip()
                        mcp_res = await session.call_tool(fname_norm, fargs)
                        preview = str(mcp_res.content)
                        log(f"✅ {preview[:600]}{'…' if len(preview)>600 else ''}")
                        
                        def _to_records_json(raw: str) -> str:
                            try:
                                data = json.loads(raw)          # mcp servers often return JSON string
                            except Exception:
                                return "[]"
                            # accept list[dict] or dict with 'rows'/'data'
                            if isinstance(data, list):
                                return json.dumps(data)
                            if isinstance(data, dict):
                                for k in ("rows", "data", "result", "content"):
                                    if k in data and isinstance(data[k], list):
                                        return json.dumps(data[k])
                            return "[]"

                        parsed = _to_records_json(preview)
                        st.session_state["last_df_json"] = parsed
                        st.session_state.setdefault("datasets", {})
                        # stable key (timestamp or hash-of-sql)
                        df_key = f"df_{int(time.time())}"
                        st.session_state["datasets"][df_key] = parsed
                        log(f"📦 Stored dataset key: {df_key}")

                        runtime_messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": fname_norm,
                            "content": preview,
                        })
                    except Exception:
                        runtime_messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": f"Error calling tool {fname_norm}: {traceback.format_exc()}",
                        })
                        runtime_messages.append({
                            "role": "assistant",    
                            "content": f"Error during tool {fname_norm} execution: {traceback.format_exc()}",
                        })
                    continue

            # 3) follow-up call
            response = client.chat.completions.create(
                model=deployment,
                messages=runtime_messages,
                tools=openai_tools,
                tool_choice="auto",
            )

        # Final assistant response
        final_text = response.choices[0].message.content or ""
        st.session_state.messages.append({"role": "assistant", "content": final_text})
        log("🎉 Turn complete")
        return final_text


# =========================
# UI: chat messages + input
# =========================
init_chat()
st.caption("Ask questions about your Postgres data. The assistant may call tools to run SQL via MCP.")

# render chat (skip system)
for m in [m for m in st.session_state.messages if m["role"] != "system"]:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m.get("content", ""))

if prompt := st.chat_input("Give me a good overview of the data you have access to."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking with tools…"):
            answer = asyncio.run(chat_turn()) # (chat_turn_langgraph()) --- does not function yet ----
            st.markdown(clean_model_output(answer) if answer.strip() else "_(no text in final reply)_")


with st.expander("Developer Console (MCP Logs)"):
    st.code("\n".join(st.session_state.logs) or "No logs yet.")

