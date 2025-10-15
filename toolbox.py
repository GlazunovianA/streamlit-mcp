import io, sys, textwrap, contextlib, traceback, time, json as _json, re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

INLINE_JSON_MAX_BYTES = 1_500_000  # ~1.5MB cap for dataframe_json

def _strip_imports(src: str) -> str:
    # Remove lines like "import x", "from x import y" (avoid ImportError since __import__ is blocked)
    lines = src.splitlines()
    cleaned = [ln for ln in lines if not re.match(r'^\s*(import\s+|from\s+\S+\s+import\s+)', ln)]
    return "\n".join(cleaned)

def handle_st_exec(args: dict):
    code = args.get("code", "")
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

    # --- Build safe globals/locals
    allowed_builtins = {
        "print": print, "len": len, "range": range, "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
        "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "any": any, "all": all, "sorted": sorted,
        "__import__": __import__, "print": print
    }
    g = {"__builtins__": allowed_builtins}
    l = {"st": st, "pd": pd, "np": np, "plt": plt, "px": px, "df": df}

    # --- Sanitize code (remove imports)
    safe_code = _strip_imports(textwrap.dedent(code))

    out, err = io.StringIO(), io.StringIO()
    ok = True
    start = time.time()

    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            # NOTE: This still runs in-process; true timeouts require multiprocessing.
            exec(safe_code, g, l)
            fig = l.get("fig", None)
            if fig is not None:
                st.pyplot(fig, clear_figure=True)
    except Exception:
        ok = False
        traceback.print_exc(file=err)

    elapsed = time.time() - start
    if elapsed > timeout_s:
        # We cannot kill the code post-fact; warn loudly.
        ok = False
        err.write(f"\nTimeout: execution took {elapsed:.2f}s > {timeout_s}s\n")

    return {"ok": ok, "stdout": out.getvalue(), "stderr": err.getvalue()}


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
                "df_key": {"type": "string", "description": "Optional: key for a dataset stored in st.session_state['datasets']"}
            },
            "required": ["code"]
        }
    }
}

