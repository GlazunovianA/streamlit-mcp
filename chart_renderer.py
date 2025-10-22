import streamlit as st


try:
  # Prefer the streamlit-chartjs component when installed
  from streamlit_chartjs.st_chart_component import st_chartjs
  HAS_ST_CHARTJS = True
except Exception:
  HAS_ST_CHARTJS = False

# Chart.js CDN HTML fallback
CHART_JS_CDN = "https://cdn.jsdelivr.net/npm/chart.js"

_HTML_TEMPLATE = '''
<div>
  <canvas id="{cid}" style="width:100%;height:400px"></canvas>
  <script src="{cdn}"></script>
  <script>
  const spec = {spec};
  const ctx = document.getElementById('{cid}').getContext('2d');
  if (window._chart_instances === undefined) window._chart_instances = {{}};
  if (window._chart_instances['{cid}']) {{ window._chart_instances['{cid}'].destroy(); }}
  window._chart_instances['{cid}'] = new Chart(ctx, spec);
  </script>
</div>
'''


def render_chart_spec(logs: str):
    """Extract Chart.js specs from logs and render them in Streamlit.

    If `streamlit-chartjs` is installed we'll call its `st_chartjs` helper. Otherwise
    we fall back to embedding a small HTML page that loads Chart.js from CDN.
    """
    import json
    if not logs:
        st.error('Empty logs received')
        return

    # Handle list input
    if isinstance(logs, list):
        logs = ''.join(str(log) for log in logs)
    else:
        logs = str(logs)
    
    # Clean up the logs string
    logs = logs.strip()
    # Remove the leading ']' if present (artifact from list formatting)
    if logs.startswith(']'):
        logs = logs[1:]
    logs = logs.strip("'\"")  # Remove any wrapping quotes
    logs = logs.replace("\\n", "\n")
    
    if '[chartjs]' not in logs:
        st.error('No chartjs spec found in logs')
        return

    # Split logs into individual chart specs and clean them
    chart_specs = []
    for raw_spec in logs.split('[chartjs]'):
        # Skip empty specs, list artifacts, and whitespace
        raw_spec = raw_spec.strip().strip("[]'\"")
        if not raw_spec or raw_spec == ',' or raw_spec.isspace():
            continue
        
        try:
            # Clean up the spec string
            spec = raw_spec.rstrip("']").rstrip("\n").rstrip("\\n").strip()
            json_obj = json.loads(spec)
            chart_specs.append(spec)
        except json.JSONDecodeError as e:
            # Only show error for non-empty specs that aren't just list artifacts
            if raw_spec and not raw_spec.startswith("[") and not raw_spec.endswith("]"):
                st.error(f"Failed to parse chart spec: {str(e)}")
    
    # Process each cleaned spec
    for payload in chart_specs:
        try:
            spec = json.loads(payload)
            
            chart_type = spec.get('type', 'bar')
            data = spec.get('data', {})
            options = spec.get('options', {})

            # Try streamlit-chartjs component first
            if HAS_ST_CHARTJS:
                try:
                    title = None
                    if isinstance(options, dict):
                        title = options.get('plugins', {}).get('title', {}).get('text')
                    st_chartjs(data=data, chart_type=chart_type, title=title)
                    continue  # Move to next chart if successful
                except Exception as e:
                    st.warning(f'st_chartjs failed, falling back to HTML embed: {e}')

            # Fallback: embed Chart.js directly
            try:
                cid = f"chart_{abs(hash(payload)) % (10**8)}"
                html = _HTML_TEMPLATE.format(
                    cid=cid, 
                    cdn=CHART_JS_CDN, 
                    spec=json.dumps(spec)
                )
                st.components.v1.html(html, height=450)
            except Exception as e:
                st.error(f"Failed to render chart spec: {e}")
                st.code(payload, language='json')

        except json.JSONDecodeError as e:
            st.error(f'Invalid JSON in chart spec: {str(e)}')
            st.code(payload, language='json')
        except Exception as e:
            st.error(f'Error processing chart spec: {str(e)}')
            st.code(payload, language='json')