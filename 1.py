#使用 ConversationSummaryBufferMemory
#ConversationSummaryBufferMemory，即对话总结缓冲记忆，它是一种混合记忆模型，结合了上述各种记忆机制，
#包括 ConversationSummaryMemory 和 ConversationBufferWindowMemory 的特点。
#这种模型旨在在对话中总结早期的互动，同时尽量保留最近互动中的原始内容。
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from LLM_API import ChatZhipuAI
from config import zhipuai_api_key

LLM = ChatZhipuAI(
temperature=0.95,# 这个值越大生成内容越随机，多样性更好
top_p=0.95,# 单步累计采用阈值，越大越多token会被考虑
api_key=zhipuai_api_key,
model_name="glm-4",
)
# 初始化对话链
conversation = ConversationChain(
    llm=LLM,
    memory=ConversationSummaryBufferMemory
    (
        llm=LLM,
        max_token_limit=300
    )
)
# 第一天的对话
# 回合1
result = conversation("我姐姐明天要过生日，我需要一束生日花束。")
print(result)
# 回合2
result = conversation("\n她喜欢粉色玫瑰，颜色是粉色的。")

# 第二天的对话
# 回合3
result = conversation("\n我又来了，还记得我昨天为什么要来买花吗？")
print(result)
