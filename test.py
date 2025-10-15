import os
from openai import AzureOpenAI

if __name__ == '__main__':

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

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "I am going to Paris, what should I see?",
            }
        ],
        max_tokens=16384,
        temperature=1.0,
        top_p=1.0,
        model=deployment
    )

    print(response.choices[0].message.content)