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

from toolbox import ST_EXEC_TOOL, handle_st_exec
from toolbox import DECISION_TOOL
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

TOOL CALLING RULES (hard)
- If you decide to use a tool, your assistant message MUST have empty `content` and include a tool call object. Do NOT describe the call in text.
- Never print pseudo-calls like “[query] …”, “{ function: … }”, or raw SQL/JSON in the chat.
- Do not say you used a tool unless there is a tool call in THIS turn.
- If a tool is relevant, call it FIRST. Do not write a final answer before tool results arrive.

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
- If a chart helps, set "viz": true in DECISION, then use tool `st_exec` to produce a Chart.js specification via the sandbox helpers.
- Use the sandbox helpers `df_to_datasets(df, x_col, y_cols, ...)` and `create_chartjs_spec(type, data, options)` to emit a standardized Chart.js spec.
- The sandbox must print the spec prefixed with the marker `[chartjs]` so the host app can detect and render it.
- The DataFrame from your last query is available as `df`. Prefer creating a small, well-labeled spec and avoid huge payloads.
- Example: single-series line chart
    ```python
    data = df_to_datasets(df, x_col='date', y_cols='value', labels='Revenue')
    create_chartjs_spec('line', data, {
            'responsive': True,
            'plugins': {'title': {'display': True, 'text': 'Revenue Over Time'}}
    })
    ```

- Example: multi-series bar chart
    ```python
    data = df_to_datasets(df, x_col='category', y_cols=['sales','profit'], labels=['Sales','Profit'])
    create_chartjs_spec('bar', data, {'plugins': {'title': {'display': True, 'text': 'Sales vs Profit'}}})
    ```

- Guidelines:
    - Keep default LIMITs (e.g., LIMIT 100) to avoid large payloads.
    - Use concise labels and numeric types for dataset values.
    - If a chart would be too large or sensitive, ask a clarifying question instead of returning a spec.

ERRORS & RETRIES
- On SQL error: fix and retry once. If it still fails, report the error briefly and ask how to proceed.
- On permission or connectivity errors: report and stop.
- On ambiguous requests: ask a clarifying question before querying.
- If the user asks to confirm, re-run the relevant tool instead of relying on previous results.

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
Assistant: 
Here are the top 10 customers by revenue. Want to drill into a specific customer?

User: “Get rows from table foo.”
Assistant:
DECISION {"need_db": true, "goal": "Verify table then sample", "sql": "", "viz": false}
[query] SELECT to_regclass('public.foo') IS NOT NULL AS exists;
Assistant: 
The table `foo` does not exist. Here are available tables: …

User: “Get rows from table foo.”
Assistant:
DECISION {"need_db": true, "goal": "Verify table then sample", "sql": "", "viz": false}
[query] SELECT to_regclass('public.foo') IS NOT NULL AS exists;
Assistant: 
The table `foo` does not exist. Here are available tables: …
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
    openai_tools.extend((ST_EXEC_TOOL, DECISION_TOOL))

    log(f"🧰 Tools available: {[t['function']['name'] for t in openai_tools]}")
    # also log full function schemas to help debugging
    logger.debug(f"OpenAI function defs: {openai_tools}")
    return openai_tools

# =========================
# Utilities for force tool use
# =========================
def needs_tool(user_text: str) -> bool:
    if not user_text:
        return False
    # Heuristic: DB-related queries should trigger tools
    k = ("select ", " from ", "table", "schema", "row", "count", "top", "average", "avg", "plot", "chart")
    t = user_text.lower()
    return any(s in t for s in k)

def looks_like_pseudo_call(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    return ("[query]" in t) or ("DECISION {" in t) or t.startswith("SELECT ") or ('{"sql":' in t)
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

        preferred_choice = {"type": "function", "function": {"name": "decision_record"}}
        # First call
        response = client.chat.completions.create(
            model=deployment,
            messages=runtime_messages,
            tools=openai_tools,
            tool_choice=preferred_choice,  # nudge: plan first
            parallel_tool_calls=True,
        )

        msg = response.choices[0].message



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
                force_next = None
                force_args = None
                if fname_norm == "decision_record":
                    # no-op: log/ack; don't show to user
                    log(f"🧭 decision_record: {fargs}")
                    runtime_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": "decision_record",
                        "content": json.dumps({"ok": True})
                    })
                    if isinstance(fargs, dict):
                        if fargs.get("next_tool") in ("query","st_exec"):
                            force_next = fargs["next_tool"]
                            if isinstance(fargs.get("next_args"), dict):
                                force_args = fargs["next_args"]
                        else:
                            # fallback heuristics from the plan
                            if fargs.get("need_db"):
                                force_next = "query"
                                if fargs.get("sql"):
                                    force_args = {"sql": fargs["sql"]}
                            elif fargs.get("viz"):
                                force_next = "st_exec"
                    continue
                if fname_norm == "st_exec":
                    try:
                        result = handle_st_exec(fargs)              # renders directly in Streamlit
                        if result["ok"] and "[chartjs]" in result["logs"]:
                            from chart_renderer import render_chart_spec
                            render_chart_spec(result["logs"])
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
                        # fall back to MCP tool call
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
                        log(f"📦 Stored dataset key : {df_key}")

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

            tool_choice_next = "auto"
            if force_next in ("query","st_exec"):
                tool_choice_next = {"type":"function","function":{"name": force_next}}
                # (Optional) give the model a gentle hint about args (it often echoes them)
                if force_args:
                    runtime_messages.append({
                        "role":"system",
                        "content": f"Next tool to call: {force_next} with args: {json.dumps(force_args)}"
                    })
            # 3) follow-up call
            response = client.chat.completions.create(
                model=deployment,
                messages=runtime_messages,
                tools=openai_tools,
                tool_choice=tool_choice_next,
                parallel_tool_calls=True,
            )

        # Final assistant response
        def _clean(text: str) -> str:
            import re
            return re.sub(r"^DECISION\s*\{.*\}\s*$", "", text or "", flags=re.MULTILINE).strip()

        final_text = _clean(response.choices[0].message.content) or ""
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
            answer = asyncio.run(chat_turn()) 
            st.markdown(clean_model_output(answer) if answer.strip() else "_(no text in final reply)_")

if st.button("😡 Urge LLM to use tools"):
    st.session_state.messages.append({"role": "user", "content": "Please make sure to use the available tools to assist with my request."})
    with st.chat_message("user"):
        st.markdown("Please make sure to use the available tools to assist with my request.")
    with st.chat_message("assistant"):
        with st.spinner("Encouraging the model to use tools…"):
            answer = asyncio.run(chat_turn()) 
            st.markdown(clean_model_output(answer) if answer.strip() else "_(no text in final reply)_")

with st.expander("Developer Console (MCP Logs)"):
    st.code("\n".join(st.session_state.logs) or "No logs yet.")

