import asyncio
import json

from openai import AzureOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Azure OpenAI client setup
endpoint = "https://az-cbc-gpt4o.openai.azure.com/"
model_name = "gpt-5-chat"
deployment = "gpt-5-chat"
subscription_key = "cfa94b27dba448069b5e64c1afd4ca68"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

async def query_with_mcp(question: str):
    # Connect to MCP Postgres server
    server_params = StdioServerParameters(
        command="docker",
        args=[
            "run","-i","--rm",
            "--add-host=host.docker.internal:host-gateway",  # 👈 add this
            "mcp-postgres",
            "postgresql://testuser:testpass@host.docker.internal:5432/testdb",
        ],
    )


    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()

            # Convert MCP tools to OpenAI function format
            openai_tools = []
            for tool in tools_result.tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })

            messages = [
                {"role": "system", "content": "You are a helpful assistant with access to a SQL database. Use the available tools to query the database when needed."},
                {"role": "user", "content": question}
            ]

            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=openai_tools,
                tool_choice="auto"
            )

            # Tool-calling loop
            while response.choices[0].message.tool_calls:
                messages.append(response.choices[0].message)

                for tool_call in response.choices[0].message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    print(f"🔧 Calling MCP tool: {function_name}")
                    print(f"📝 Arguments: {function_args}")

                    mcp_result = await session.call_tool(function_name, function_args)
                    print(f"✅ MCP Result: {mcp_result.content}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(mcp_result.content)
                    })

                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto"
                )

            return response.choices[0].message.content

async def main():
    question = "Give me a good overview of the data."
    answer = await query_with_mcp(question)
    print(f"\n🤖 Final Answer: {answer}")

asyncio.run(main())
