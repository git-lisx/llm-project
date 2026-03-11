from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage

# ChatTongyi不支持自定义模型部署的 endpoint URL
chatLLM = ChatTongyi(
    model="qwen-plus",
    model_kwargs={
        "enable_thinking":True
    }
)
completion = chatLLM.stream([HumanMessage(content="你是谁")])
is_answering = False
print("="*20+"思考过程 "+"="*20)
for chunk in completion:
    if chunk.additional_kwargs.get("reasoning_content"):
        print(chunk.additional_kwargs.get("reasoning_content"),end="",flush=True)
    else:
        if not is_answering:
            print("\n"+"="*20+"回复内容"+"="*20)
            is_answering = True
        print(chunk.content,end="",flush=True)