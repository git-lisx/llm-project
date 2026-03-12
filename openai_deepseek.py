# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

# stream = False
stream = True

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你是谁？"},
    ],
    stream=stream
)

if stream:
    for chunk in response:
        delta = chunk.choices[0].delta
        if not delta or delta.content is None:
            continue
        print(delta.content, end="", flush=True)
    print()
else:
    print(response.choices[0].message.content)

print()
