import streamlit as st
from streamlit_chat import message as st_message  # 引入streamlit-chat的message组件
# from st_chat_message import message as st_message
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain_community.vectorstores import FAISS
from LoadVector import DocumentService
import pandas as pd
from LLM_API import ChatZhipuAI
from config import zhipuai_api_key,vector_store_path,qianfan_ak,qianfan_sk,model_embed,endpoint_embed
from zhipuai import ZhipuAI
from langchain_community.llms import QianfanLLMEndpoint
import os

@st.cache_resource
def LLM_API(user_query,system_prompt):
    client = ZhipuAI(api_key=zhipuai_api_key)
    response = client.chat.completions.create(
    model="glm-4",  # 填写需要调用的模型名称
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ],
    )
    return response.choices[0].message.content

@st.cache_resource
def init_LLM():
    LLM = ChatZhipuAI(
    temperature=0.95,# 这个值越大生成内容越随机，多样性更好
    top_p=0.95,# 单步累计采用阈值，越大越多token会被考虑
    api_key=zhipuai_api_key,
    model_name="glm-4",
    )

    os.environ["QIANFAN_AK"] = qianfan_ak
    os.environ["QIANFAN_SK"] = qianfan_sk
    LLM = QianfanLLMEndpoint(model='ERNIE-Bot-4',temperature=0.8)
    return LLM

#@st.cache_data
def init_data(vector_store_path,docs_paths):
    # print(docs_paths)
    vector_store_path=vector_store_path
    # EMBEDDING_model_dir=EMBEDDING_model_dir
    doc_service = DocumentService(docs_paths,vector_store_path)
    doc_service.init_source_vector()

    return doc_service

def prompt_template(system):
    str_system=f"你是一个{system}小助手，你的任务是为用户提供专业、准确、有见地的建议。"
    prompt_template = str_system+"""请根据已知内容:
                                    {context}
                                    回答下面问题:
                                    {question}"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt

def init_RetrievalQA(LLM,doc_service,prompt):
    
    knowledge_chain = RetrievalQA.from_llm(llm=LLM,retriever=doc_service.load_vector_store(),prompt=prompt)
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(input_variables=["page_content"], template="{page_content}")
    
    return knowledge_chain

# Streamlit应用的主要部分
def main():

    st.sidebar.header("**本地知识库加载**")
    
    # 当accept_multiple_files=True，返回的uploaded_files为一个list
    uploaded_files = st.sidebar.file_uploader("选择文档文件", type=['txt','md'], accept_multiple_files=True,help="支持txt、csv、md文件格式")

    st.title("智能小助手")
    # 使用streamlit-chat来管理对话历史
    if 'dialogues' not in st.session_state:
        st.session_state.dialogues = []
    docs_paths = []  # 存储所有文档路径的列表
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "text/plain" or uploaded_file.name.endswith('.md'):
            # 处理文本文件，这里简单示例合并文本内容到一个文件
            docs_content = uploaded_file.read()
            if isinstance(docs_content, bytes):
                docs_content = docs_content.decode('utf-8')
            text_path = f'{vector_store_path}{uploaded_file.name}'
            with open(text_path, 'w', encoding='utf-8') as file:
                file.write(docs_content)
            docs_paths.append(text_path)

    system="推免"
    if len(uploaded_files) == 0:
        knowledge_chain = LLM_API
    else:
        LLM=init_LLM()
        doc_service=init_data(vector_store_path,docs_paths)
        prompt=prompt_template(system)
        knowledge_chain = init_RetrievalQA(LLM,doc_service,prompt)

    # 对话显示和输入逻辑改为使用streamlit-chat
    # 显示对话历史
    # for dialogue in st.session_state.dialogues:
    #     st_message(**dialogue, key=f"dialogue_{int(time.time() * 1000)}")

    for i, dialogue in enumerate(st.session_state.dialogues):
        # 使用循环的索引i来生成唯一的 key
        st_message(**dialogue, key=f"dialogue_{i}")

        
    user_query = st.chat_input("请输入问题：")
    if user_query:
        # 假设知识链已经被正确初始化
        result_txt = "无法找到答案，请稍后再试。"
        if len(uploaded_files) == 0:
            system_prompt=f"你是一个{system}小助手，你的任务是为用户提供专业、准确、有见地的建议。"
            result_txt = knowledge_chain(user_query,system_prompt)
        else:
            result = knowledge_chain.invoke({"query": user_query})
            result_txt = result['result']
        # 更新对话历史，使用streamlit-chat格式
        st.session_state.dialogues.append({"message": user_query, "is_user": True})
        st.session_state.dialogues.append({"message": result_txt, "is_user": False})
        # 触发页面重新运行来更新对话显示
        st.rerun()

if __name__ == "__main__":
    main()
