import os
import json
from e2b_code_interpreter import Sandbox

def run_chartjs_test():
    os.environ.setdefault('E2B_API_KEY', os.environ.get('E2B_API_KEY', ''))
    sbx = Sandbox.create()
    code = """
import pandas as pd

# sample data
df = pd.DataFrame({'category': ['A','B','C'], 'value': [10,20,30]})

def df_to_datasets(df, x_col, y_cols, labels=None):
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    if labels is None:
        labels = y_cols
    return {
        'labels': df[x_col].tolist(),
        'datasets': [
            {
                'label': labels[i],
                'data': df[col].tolist(),
            }
            for i, col in enumerate(y_cols)
        ]
    }

data = df_to_datasets(df, x_col='category', y_cols='value', labels=['Value'])
spec = {'type': 'bar', 'data': data, 'options': {'plugins': {'title': {'display': True, 'text': 'Test Chart'}}}}
print('[chartjs]' + json.dumps(spec))
"""
    execution = sbx.run_code(code)
    # Print logs and outputs
    logs = getattr(execution, 'logs', None) or getattr(execution, 'stdout', None) or ''
    print('=== SANDBOX LOGS ===')
    print(logs)
    print('=== END LOGS ===')

if __name__ == '__main__':
    run_chartjs_test()