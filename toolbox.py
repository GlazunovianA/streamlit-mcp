import os, io, time, mimetypes, sys, textwrap, contextlib, traceback, time, json, re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox

INLINE_JSON_MAX_BYTES = 1_500_000  # ~1.5MB cap for dataframe_json

def _strip_imports(src: str) -> str:
    # Remove lines like "import x", "from x import y" (avoid ImportError since __import__ is blocked)
    lines = src.splitlines()
    cleaned = [ln for ln in lines if not re.match(r'^\s*(import\s+|from\s+\S+\s+import\s+)', ln)]
    return "\n".join(cleaned)

def handle_st_exec(args: dict):
    code = args.get("code", "")
    load_dotenv()
    os.environ["E2B_API_KEY"] = args.get("e2b_api_key").replace("\r", "").replace("\n", "").strip().strip('"').strip("'")
    # -------------- sanity check --------------
    key = args.get("e2b_api_key") or os.getenv("E2B_API_KEY") or st.secrets.get("E2B_API_KEY")
    if not key:
        return {"ok": False, "logs": "", "error": "Missing E2B_API_KEY", "elapsed": 0.0}

    INLINE_JSON_MAX_BYTES = 2_000_000  # ~2MB
    sbx = Sandbox.create() # By default the sandbox is alive for 5 minutes
    # ----------- TEST SANDBOX -----------
    execution = sbx.run_code("print('hello world')") # Execute Python inside the sandbox
    print(execution.logs)

    execution = sbx.run_code(code)
    print(execution.logs)
    # ------------------------------------


    files = sbx.files.list("/")
    print(files)
    print('------------- execution successful! ------------')
    timeout_s = int(args.get("timeout_s", 5))

    # --- Prepare df from (1) dataframe_json -> (2) df_key -> (3) last_df_json
    df = None

    dataframe_json = args.get("dataframe_json")
    if dataframe_json:
        try:
            # size guard
            if isinstance(dataframe_json, str) and len(dataframe_json.encode("utf-8")) > INLINE_JSON_MAX_BYTES:
                raise ValueError("dataframe_json too large; pass df_key instead.")
            df = pd.read_json(io.StringIO(dataframe_json), orient="records")
        except Exception as e:
            df = None
            st.warning(f"Failed to parse dataframe_json: {e}")

    if df is None and args.get("df_key"):
        datasets = st.session_state.get("datasets") or {}
        j = datasets.get(args["df_key"])
        if j:
            try:
                df = pd.read_json(io.StringIO(j), orient="records")
            except Exception as e:
                st.warning(f"Failed to load df from df_key={args['df_key']}: {e}")

    if df is None:
        j = st.session_state.get("last_df_json")
        if j:
            try:
                df = pd.read_json(io.StringIO(j), orient="records")
            except Exception as e:
                st.warning(f"Failed to load df from last_df_json: {e}")


    prelude_lines = [
        "import os, json, pandas as pd",
        "from pathlib import Path",
        "ARTIFACTS_DIR = '/artifacts'",
        "Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)",
        "",
        "def save_text(text, name='output.txt'):",
        "    p = os.path.join(ARTIFACTS_DIR, name)",
        "    with open(p, 'w', encoding='utf-8') as f: f.write(text)",
        "    print(f'[artifact] text:{name}')",
        "    return p",
        "",
        "def save_table(df, name='table.csv'):",
        "    p = os.path.join(ARTIFACTS_DIR, name)",
        "    df.to_csv(p, index=False)",
        "    print(f'[artifact] table:{name}')",
        "    return p",
        "",
        "def save_json(obj, name='data.json'):",
        "    p = os.path.join(ARTIFACTS_DIR, name)",
        "    with open(p, 'w', encoding='utf-8') as f: json.dump(obj, f, ensure_ascii=False)",
        "    print(f'[artifact] json:{name}')",
        "    return p",
        "",
        "def save_fig(fig=None, name='figure.png', dpi=150):",
        "    import matplotlib.pyplot as plt",
        "    fig = fig or plt.gcf()",
        "    p = os.path.join(ARTIFACTS_DIR, name)",
        "    fig.savefig(p, dpi=dpi, bbox_inches='tight')",
        "    print(f'[artifact] image:{name}')",
        "    return p",
    ]

    # inject df if present
    if df is not None:
        try:
            df_json_str = df.to_json(orient="records")
            prelude_lines += [
                f"DF_JSON = r'''{df_json_str}'''",
                "df = pd.read_json(DF_JSON, orient='records')",
            ]
        except Exception as e:
            st.warning(f"Failed to serialize df for sandbox: {e}")

    prelude_code = "\n".join(prelude_lines)

    wrapped_user_code = f"""
# --- prelude ---
{prelude_code}

# --- user code ---
{code}
"""

    # ---------- run in sandbox ----------
    ok, logs, error, artifacts, start = True, "", "", [], time.time()
    sbx = Sandbox.create()
    try:
        execution = sbx.run_code(wrapped_user_code, timeout=timeout_s)
        logs = getattr(execution, "logs", "") or ""
        err = getattr(execution, "error", "") or getattr(execution, "traceback", "")
        if err:
            ok = False
            error = str(err)

        # pull artifacts back
        try:
            entries = sbx.files.list("/artifacts")  # [{'path': '/artifacts/..', 'type': 'file', ...}, ...]
        except Exception:
            entries = []

        for ent in entries or []:
            if ent.get("type") != "file":
                continue
            path = ent.get("path")
            try:
                content = sbx.files.read(path)  # bytes
                mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
                artifacts.append({"path": path, "bytes": content, "mime": mime})
            except Exception:
                # ignore unreadable entries
                pass
    except Exception as e:
        ok, error = False, f"{type(e).__name__}: {e}"
    finally:
        elapsed = time.time() - start
        if elapsed > timeout_s:
            ok = False
            error = (error + "\n" if error else "") + f"Timeout: execution took {elapsed:.2f}s > {timeout_s}s"
        try:
            sbx.close()
        except Exception:
            pass

    return {"ok": ok, "logs": logs, "error": error, "elapsed": elapsed, "artifacts": artifacts}

ST_EXEC_TOOL = {
    "type": "function",
    "function": {
        "name": "st_exec",
        "description": "Execute short Python to render charts/tables in Streamlit. Use st.* APIs; may also use matplotlib.pyplot as plt or plotly.express as px.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code that calls st.* or produces a fig"},
                "timeout_s": {"type": "integer", "default": 5},
                "dataframe_json": {"type": "string", "description": "Optional: records-oriented JSON for df (small payloads only)"},
                "df_key": {"type": "string", "description": "Optional: key for a dataset stored in st.session_state['datasets']"},
                "e2b_api_key": {"type": "string", "description": "E2B API key to access E2B services"},
            },
            "required": ["code", "e2b_api_key"]
        }
    }
}

