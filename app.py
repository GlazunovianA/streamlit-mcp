import asyncio
import json
import os
from contextlib import asynccontextmanager
import time

import streamlit as st
from openai import AzureOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


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

    # Deployment name to use in chat.completions.create(model=...)
    deployment = st.text_input(
        "Deployment (model) name",
        value=st.secrets.get("azure", {}).get("deployment", "gpt-5-chat"),
    )


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
    "You are a helpful assistant for a **PostgreSQL** database."
    "Use only the provided tools. Produce valid PostgreSQL SQL."
    "You do not have to ask for permission for executing query code. all SQL is executed in a controlled sandbox."
    "When you need to inspect or compute data, always call the query tool directly, even for dynamic or multi-table SQL."
    "Do not print JSON or SQL in the reply text, but rather call the tool."
    "If information is insufficient, ask clarifying questions."
    "To render charts, you may call the st_exec tool and write concise Python that uses st.line_chart, st.bar_chart, st.scatter_chart, st.pyplot (with matplotlib.pyplot as plt), or st.plotly_chart (with plotly.express as px). If you use Matplotlib, assign the figure to fig or call st.pyplot(fig)."
    "When you need data or to render charts, use tool calls. Do not print JSON function calls in your reply content."
    "After you call query, the resulting rows are available as a pandas DataFrame named df inside st_exec code."
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

""" # =========================
# Docker args builder
# =========================
def build_docker_args():
    args = ["run", "-i", "--rm"]
    if add_host_gateway:
        args += ["--add-host=host.docker.internal:host-gateway"]
    if use_host_network:
        args += ["--network=host"]
    if network.strip():
        args += ["--network", network.strip()]
    # image + DB URI as final args
    args += [docker_image, db_uri]
    return args
 """
# =========================
# MCP session (async CM)
# =========================
@asynccontextmanager
async def mcp_session():
    # 1) Validate the binary exists
    if not os.path.exists(mcp_path):
        st.error(f"MCP server not found at: {mcp_path}")
        raise FileNotFoundError(mcp_path)

    # 2) Build args for your MCP server
    # If your MCP server accepts a single Postgres URI as the only arg:
    server_args = [db_uri]

    env = None

    server_params = StdioServerParameters(
        command=mcp_path,
        args=server_args,
        env=env,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session

# =========================
# Tools bridge
# =========================


async def get_openai_tools(session: ClientSession):
    tools_result = await session.list_tools()
    tools = tools_result.tools
    log(f"🧩 Raw tool schemas: {[getattr(t, 'inputSchema', {}) for t in tools]}")

    names = [t.name for t in tools]
    log(f"🧰 Discovered tools: {names}")

    openai_tools = [{
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.inputSchema,
        },
    } for t in tools]

    # keep your custom exec tool
    openai_tools.append(ST_EXEC_TOOL)

    log(f"🧰 Tools available: {[t['function']['name'] for t in openai_tools]}")
    return openai_tools


# =========================
# One chat turn with tools
# =========================
async def chat_turn():
    st.session_state.logs.clear()

    # Only system/user/assistant in the long-term memory
    runtime_messages = [
        m for m in st.session_state.messages
        if m["role"] in ("system", "user", "assistant")
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
            log(f"Received tool_calls: {response.choices[0].message.tool_calls!r}")
            
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

                # ✅ normalize common aliases the model might emit
                alias = {
                    "functions.query": "query",
                    "functions.st_exec": "st_exec",
                    # add anything else the model might invent
                }
                fname_norm = alias.get(fname, fname)

                log(f"🔧 {fname} → {fname_norm} | 📝 {fargs}")

                if fname == "st_exec":
                    result = handle_st_exec(fargs)              # renders directly in Streamlit
                    runtime_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    })
                    continue
                else:
                    mcp_res = await session.call_tool(fname, fargs)
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
                    if "datasets" not in st.session_state:
                        st.session_state["datasets"] = {}
                    # stable key (timestamp or hash-of-sql)
                    ds_key = f"ds_{int(time.time())}"
                    st.session_state["datasets"][ds_key] = parsed
                    log(f"📦 Stored dataset key: {ds_key}")

                runtime_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": preview,
                })

            # 3) follow-up call
            response = client.chat.completions.create(
                model=deployment,
                messages=runtime_messages,
                tools=openai_tools,
                tool_choice="auto",
            )

        # Final assistant response
        final_text = response.choices[0].message.content
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
            st.markdown(answer if answer.strip() else "_(no text in final reply)_")


with st.expander("Developer Console (MCP Logs)"):
    st.code("\n".join(st.session_state.logs) or "No logs yet.")




'''I would like to know what weather_data contains. can you visualize something about it?'''