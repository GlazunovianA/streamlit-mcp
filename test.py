import os
from openai import AzureOpenAI

if __name__ == '__main__':

    os.environ["E2B_API_KEY"] = "e2b_a1188ca1bbb0e65ac943149b41520511a9ebe0a5"
    from e2b_code_interpreter import Sandbox
    Sandbox.create()  # should succeed; if not, the key itself is invalid/expired
