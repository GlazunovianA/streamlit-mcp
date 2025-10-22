import os, io, time, mimetypes, re
import json as _json
import pandas as pd
from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox
import base64, binascii

INLINE_JSON_MAX_BYTES = 2_000_000  # ~2MB

def handle_st_exec(args: dict):
    import streamlit as st
    """
    Execute Python in an e2b sandbox, collect artifacts saved under /artifacts,
    render them in Streamlit, and return a JSON-safe summary (no bytes).

    Args:
      - code: str (required)                   # user code to run in sandbox
      - timeout_s: int (default 5)
      - dataframe_json / df_key / last_df_json: optional DataFrame to inject as `df`
      - e2b_api_key: str (optional; takes precedence over env/secrets)

    Returns (JSON-safe):
      {
        "ok": bool,
        "logs": str,
        "error": str,
        "elapsed": float,
        "artifacts": [{"path": str, "mime": str, "size": int}]
      }
    """
    load_dotenv()

    key = st.secrets.get("e2b", {}).get("e2b_api_key")
    
    key = key.replace("\r", "").replace("\n", "").strip().strip('"').strip("'")

    if not key:
        st.error("Missing E2B_API_KEY. Add to .streamlit/secrets.toml or env.")
        return {"ok": False, "logs": "", "error": "Missing E2B_API_KEY", "elapsed": 0.0, "artifacts": []}

    os.environ["E2B_API_KEY"] = key

    # ---------------------------
    # Optional: resolve a DataFrame for injection as `df`
    # ---------------------------
    def _json_to_df(j: str):
        return pd.read_json(io.StringIO(j), orient="records")

    df = None
    j = args.get("dataframe_json")
    if j:
        try:
            if isinstance(j, str) and len(j.encode("utf-8")) > INLINE_JSON_MAX_BYTES:
                raise ValueError("dataframe_json too large; pass df_key instead.")
            df = _json_to_df(j)
        except Exception as e:
            st.warning(f"Failed to parse dataframe_json: {e}")

    if df is None and args.get("df_key"):
        datasets = st.session_state.get("datasets") or {}
        j = datasets.get(args["df_key"])
        if j:
            try:
                df = _json_to_df(j)
            except Exception as e:
                st.warning(f"Failed to load df from df_key={args['df_key']}: {e}")

    if df is None:
        j = st.session_state.get("last_df_json")
        if j:
            try:
                df = _json_to_df(j)
            except Exception as e:
                st.warning(f"Failed to load df from last_df_json: {e}")

    # ---------------------------
    # Prelude that defines Chart.js visualization helpers for the sandbox
    # ---------------------------
    prelude_lines = [
    "import os, sys, pandas as pd, types",
    "import json",
    "",
    "def create_chartjs_spec(type='bar', data=None, options=None):",
    "    '''Create and output a Chart.js specification'''",
    "    spec = {",
    "        'type': type,",
    "        'data': data or {},",
    "        'options': options or {}",
    "    }",
    "    print('[chartjs]' + json.dumps(spec))",
    "",
    "def df_to_datasets(df, x_col, y_cols, labels=None, colors=None):",
    "    '''Convert DataFrame columns to Chart.js datasets'''",
    "    if isinstance(y_cols, str):",
    "        y_cols = [y_cols]",
    "    if labels is None:",
    "        labels = y_cols",
    "    if colors is None:",
    "        colors = ['rgb(75, 192, 192)'] * len(y_cols)",
    "    ",
    "    datasets = []",
    "    for i, col in enumerate(y_cols):",
    "        dataset = {",
    "            'label': labels[i] if i < len(labels) else col,",
    "            'data': df[col].tolist(),",
    "            'backgroundColor': colors[i] if i < len(colors) else 'rgb(75, 192, 192)',",
    "            'borderColor': colors[i] if i < len(colors) else 'rgb(75, 192, 192)',",
    "            'borderWidth': 1",
    "        }",
    "        datasets.append(dataset)",
    "    ",
    "    return {",
    "        'labels': df[x_col].tolist(),",
    "        'datasets': datasets",
    "    }",
    "",
    "def plot_line(df, x, y, title=None):",
    "    '''Create a line chart from DataFrame columns'''",
    "    data = df_to_datasets(df, x, y)",
    "    create_chartjs_spec('line', data, {'title': title} if title else None)",
    "",
    "def plot_bar(df, x, y, title=None):",
    "    '''Create a bar chart from DataFrame columns'''",
    "    data = df_to_datasets(df, x, y)",
    "    create_chartjs_spec('bar', data, {'title': title} if title else None)",
    "",
    "def plot_scatter(df, x, y, title=None):",
    "    '''Create a scatter chart from DataFrame columns'''",
    "    data = df_to_datasets(df, x, y)",
    "    create_chartjs_spec('scatter', data, {'title': title} if title else None)",
    "",
    "# ---- streamlit shim injected as a fake module so `import streamlit as st` works ----",
    "def _as_dataframe(obj):",
    "    if isinstance(obj, pd.DataFrame):",
    "        return obj",
    "    try:",
    "        return pd.DataFrame(obj)",
    "    except Exception:",
    "        return pd.DataFrame({'value':[obj]})",
    "",
    "class _ShimST:",
    "    def write(self, x):",
    "        save_text(str(x), name='stdout.txt')",
    "    def markdown(self, x):",
    "        save_text(str(x), name='markdown.md')",
    "    def code(self, x):",
    "        save_text(str(x), name='code.txt')",
    "    def dataframe(self, data):",
    "        df = _as_dataframe(data)",
    "        save_table(df, name='dataframe.csv')",
    "    def pyplot(self, fig=None):",
    "        save_fig(fig=fig, name='plot.png')",
    "    def _plot_df(self, data, kind, name):",
    "        import matplotlib.pyplot as plt",
    "        df = _as_dataframe(data)",
    "        # try to use the first non-index column if needed",
    "        try:",
    "            if kind == 'bar':",
    "                ax = df.plot(kind='bar')",
    "            elif kind == 'line':",
    "                ax = df.plot(kind='line')",
    "            elif kind == 'scatter':",
    "                # scatter needs x/y; pick first two columns if available",
    "                cols = list(df.columns)",
    "                if len(cols) >= 2:",
    "                    ax = df.plot(kind='scatter', x=cols[0], y=cols[1])",
    "                else:",
    "                    ax = df.plot(kind='line')",
    "            else:",
    "                ax = df.plot(kind='line')",
    "            fig = ax.get_figure()",
    "            save_fig(fig=fig, name=name)",
    "            plt.close(fig)",
    "        except Exception as _e:",
    "            save_text(f'Plot error: {type(_e).__name__}: {_e}', name='plot_error.txt')",
    "    def bar_chart(self, data):",
    "        self._plot_df(data, kind='bar', name='bar_chart.png')",
    "    def line_chart(self, data):",
    "        self._plot_df(data, kind='line', name='line_chart.png')",
    "    def scatter_chart(self, data):",
    "        self._plot_df(data, kind='scatter', name='scatter_chart.png')",
    "",
    "# create module-like object and register under sys.modules",
    "_st_mod = types.SimpleNamespace(**{k:getattr(_ShimST(), k) for k in dir(_ShimST) if not k.startswith('_')})",
    "sys.modules['streamlit'] = _st_mod",
    ]


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

    # ---------------------------
    # Wrap user code
    # ---------------------------
    user_code = args.get("code", "") or ""
    timeout_s = int(args.get("timeout_s", 5))

    wrapped_user_code = f"""
# --- prelude ---
{prelude_code}

# --- user code ---
{user_code}
"""

    # ---------------------------
    # Execute in sandbox (single sandbox)
    # ---------------------------
    ok, logs, error, artifacts_meta = True, "", "", []
    start = time.time()
    sbx = Sandbox.create()
    try:
        execution = sbx.run_code(wrapped_user_code, timeout=timeout_s)

        # Coerce logs to string (avoid "Logs is not JSON serializable")
        raw_logs = getattr(execution, "logs", "")
        if isinstance(raw_logs, str):
            logs = raw_logs
        elif isinstance(raw_logs, list):
            logs = ''.join(str(line) for line in raw_logs)
        else:
            logs = getattr(raw_logs, "text", None) or getattr(raw_logs, "stdout", None) or str(raw_logs)
        logs = str(logs).strip()

        # Chart.js specs will be handled by chart_renderer.py
            
        err = getattr(execution, "error", "") or getattr(execution, "traceback", "")
        if err:
            ok = False
            error = str(err)

        # Pull and render artifacts
        try:
            entries = sbx.files.list("/artifacts")
        except Exception:
            entries = []


        def _get(ent, attr, default=None):
            # support both dict and object entries
            if isinstance(ent, dict):
                return ent.get(attr, default)
            return getattr(ent, attr, default)


        def _maybe_b64(s: str) -> bool:
            s2 = s.strip()
            if len(s2) % 4 != 0:
                return False
            return re.fullmatch(r'[A-Za-z0-9+/=\s]+', s2) is not None

        def _read_bytes(sbx, path):
            """Normalize sbx.files.read() into raw bytes (handles bytes/str/dict/base64)."""
            try:
                # First try the new e2b API for getting files
                if hasattr(sbx, 'filesystem'):
                    content = sbx.filesystem.read(path)
                    if isinstance(content, (bytes, bytearray)):
                        return bytes(content)
                    elif isinstance(content, str):
                        return content.encode('utf-8')
                
                # Fallback to old API
                obj = sbx.files.read(path)

                # 1) already bytes
                if isinstance(obj, (bytes, bytearray)):
                    return bytes(obj)

                # 2) plain string: try base64; if not base64, try latin-1 to preserve bytes
                if isinstance(obj, str):
                    s = obj
                    if _maybe_b64(s):
                        try:
                            return base64.b64decode(s, validate=True)
                        except binascii.Error:
                            pass
                    # latin-1 roundtrip preserves codepoints 0..255 without inserting U+FFFD
                    return s.encode("latin-1", errors="replace")

                # 3) dict-like payloads some SDKs return
                if isinstance(obj, dict):
                    enc = obj.get("encoding")
                    data = obj.get("content") or obj.get("data") or obj.get("bytes")
                    if data is None:
                        return None
                    if isinstance(data, (bytes, bytearray)):
                        return bytes(data)
                    if isinstance(data, str):
                        if enc == "base64" or _maybe_b64(data):
                            try:
                                return base64.b64decode(data, validate=True)
                            except binascii.Error:
                                return data.encode("latin-1", errors="replace")
                        return data.encode("latin-1", errors="replace")

                return None
            except Exception as e:
                st.error(f"Error reading file {path}: {str(e)}")
                return None

                # (visualization handling moved to earlier in the code)

    except Exception as e:
        ok = False
        error = f"{type(e).__name__}: {e}"
        st.error(error)
    finally:
        elapsed = time.time() - start
        if elapsed > timeout_s:
            ok = False
            timeout_msg = f"Timeout: execution took {elapsed:.2f}s > {timeout_s}s"
            error = (error + "\n" if error else "") + timeout_msg
            st.warning(timeout_msg)
        try:
            sbx.close()
        except Exception:
            pass

    # Return JSON-safe summary (no bytes)
    return {
        "ok": bool(ok),
        "logs": str(logs)[:20000],
        "error": str(error) if error else "",
        "elapsed": float(elapsed),
        "artifacts": artifacts_meta,
    }

ST_EXEC_TOOL = {
    "type": "function",
    "function": {
        "name": "st_exec",
        "description": "Create Chart.js visualizations. Helper functions available: create_chartjs_spec() and df_to_datasets()",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": """Python code to create Chart.js visualizations.  Example usage:
                        import pandas as pd
                        import numpy as np
                        import json
                        def plot_line(df, x_col, y_cols, title=None):
                            data = df_to_datasets(df, x_col, y_cols)
                            options = {'title': {'text': title}} if title else {}
                            create_chartjs_spec('line', data, options)

                        def plot_bar(df, x_col, y_col, title=None):
                            data = df_to_datasets(df, x_col, y_col)
                            options = {'title': {'text': title}} if title else {}
                            create_chartjs_spec('bar', data, options)

                        def plot_scatter(df, x_col, y_col, title=None):
                            data = {
                                'labels': df[x_col].tolist(),
                                'datasets': [{
                                    'label': y_col,
                                    'data': df[y_col].tolist(),
                                    'backgroundColor': 'rgb(75, 192, 192)',
                                    'borderColor': 'rgb(75, 192, 192)',
                                    'borderWidth': 1
                                }]
                            }
                            options = {'title': {'text': title}} if title else {}
                            create_chartjs_spec('scatter', data, options)

                        # Create sample data
                        df = pd.DataFrame({
                            'x': range(5),
                            'y1': [1, 2, 3, 2, 1],
                            'y2': [2, 1, 2, 3, 2],
                            'y3': [1, 3, 5, 3, 1]
                        })

                        # Test line chart with multiple series
                        plot_line(df, 'x', ['y1', 'y2'], title='Multi-Series Line Chart')

                        # Test bar chart
                        plot_bar(df, 'x', 'y3', title='Simple Bar Chart')

                        # Test scatter plot
                        scatter_data = pd.DataFrame({
                            'x': np.random.rand(10),
                            'y': np.random.rand(10)
                        })
                        plot_scatter(scatter_data, 'x', 'y', title='Scatter Plot')
                    """,    
                    
                }
            },
            "required": ["code"]
        }
    }
}


DECISION_TOOL = {
    "type": "function",
    "function": {
        "name": "decision_record",
        "description": "Record a compact plan before acting. No user-visible output.",
        "parameters": {
            "type": "object",
            "properties": {
                "need_db": {"type": "boolean"},
                "goal":    {"type": "string"},
                "sql":     {"type": "string"},
                "viz":     {"type": "boolean"},
                "next_tool": {"type": "string", "enum":["query","st_exec"]},
                "next_args": {"type": "object"},
            },
            "required": ["need_db", "goal", "viz"]
        }
    }
}

