# pip install langchain-core langchain-openai
import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


class DeepSeekChatModel:
    def __init__(self, model_name="deepseek-chat", temperature=0.7):
        """
        初始化DeepSeek聊天模型
        :param model_name: 模型名称，默认为"deepseek-chat"
        :param temperature: 控制输出随机性的参数，值越高越随机
        """
        self.chat_model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
        )

    def chat(self, message: str):
        """
        发送消息到DeepSeek模型并返回响应
        :param message: 输入的消息字符串
        :return: 模型返回的响应内容
        """
        try:
            # 创建人类消息对象
            human_message = HumanMessage(content=message)

            # 调用模型获取响应
            response = self.chat_model.invoke([human_message])

            return response.content
        except Exception as e:
            return f"发生错误: {str(e)}"

    def reasoning_chat_with_langchain(self, message: str):
        """
        使用LangChain实现推理模式，通过额外参数启用思考模式
        :param message: 输入的消息字符串
        :return: 包含推理过程和最终答案的字典
        """
        try:
            # 使用 LangChain 的 extra_body 参数传递 thinking 选项
            response = self.chat_model.invoke(
                [HumanMessage(content=message)],
                extra_body={"thinking": {"type": "enabled"}}  # 启用思考模式
            )
            
            # 检查是否有推理内容属性
            reasoning_content = getattr(response, 'reasoning_content', None)
            final_content = response.content
            
            return {
                "reasoning": reasoning_content,
                "answer": final_content
            }
        except Exception as e:
            return {
                "reasoning": None,
                "answer": f"发生错误: {str(e)}"
            }

    def stream_chat(self, message: str):
        """
        流式返回聊天结果（如果API支持）
        :param message: 输入的消息字符串
        :return: 生成器，逐个返回响应片段
        """
        try:
            human_message = HumanMessage(content=message)

            # 使用流式调用
            for chunk in self.chat_model.stream([human_message]):
                yield chunk.content
        except Exception as e:
            yield f"发生错误: {str(e)}"


def main(stream_chat: bool = True):
    # 创建DeepSeek模型实例
    deepseek = DeepSeekChatModel(model_name="deepseek-reasoner", temperature=0.5)

    print("欢迎使用DeepSeek聊天机器人！输入'quit'退出程序。")
    user_input = "你是谁？"
    print("\nDeepSeek: ", end="")
    for chunk in deepseek.stream_chat(user_input):
        print(chunk, end="", flush=True)

    # while True:
    #     user_input = input("\n您: ")
    #
    #     if user_input.lower() == 'quit':
    #         print("再见！")
    #         break
    #
    #     if stream_chat:
    #         # 流式响应
    #         print("\nDeepSeek: ", end="")
    #         for chunk in deepseek.stream_chat(user_input):
    #             print(chunk, end="", flush=True)
    #
    #     else:
    #         # 获取模型响应
    #         response = deepseek.chat(user_input)
    #         print(f"\nDeepSeek: {response}")
    #
    #     print()


if __name__ == "__main__":
    main()