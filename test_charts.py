import streamlit as st
import pandas as pd
import numpy as np

# Test basic chart creation
st.write("# Chart.js Visualization Tests")

code = '''
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
'''

from toolbox import handle_st_exec
result = handle_st_exec({"code": code})

if result["ok"]:
    from chart_renderer import render_chart_spec
    render_chart_spec(result["logs"])
    st.write("Raw logs:")
    st.code(result["logs"])
else:
    st.error(f"Error: {result['error']}")