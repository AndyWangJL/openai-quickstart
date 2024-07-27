import gradio as gr

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain, StuffDocumentsChain
from langchain_community.vectorstores import FAISS
from langchain.prompts import (PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate,
                               HumanMessagePromptTemplate)
from utils import ArgumentParser


def initialize_sales_bot(vector_store_dir: str = "car_estates_sale"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name=args.model_name, temperature=1)

    # 自定义的 SystemMessagePromptTemplate
    custom_system_message_template = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=['context'],
            template="""Use the following pieces of context to answer the user's question. If you don’t have specific 
            information, provide practical suggestions or alternative ways to find the information in a natural and 
            friendly tone. Ensure the response feels personal and helpful, avoiding any specific examples in the 
            template itself. ---------------- {context}"""
        )
    )

    # 创建 LLMChain 并嵌入自定义的 SystemMessagePromptTemplate
    llm_chain = LLMChain(
        prompt=ChatPromptTemplate(
            input_variables=['context', 'question'],
            messages=[
                custom_system_message_template,
                HumanMessagePromptTemplate(
                    prompt=PromptTemplate(
                        input_variables=['question'],
                        template='{question}'
                    )
                )
            ]
        ),
        llm=llm
    )

    # 创建 StuffDocumentsChain 并使用自定义的 LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name='context'
    )

    global SALES_BOT
    SALES_BOT = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        retriever=db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8}),
        return_source_documents=True
    )

    print("SALES_BOT initialized:", SALES_BOT)

    return SALES_BOT


def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # 从命令行参数中获取
    enable_chat = args.enable_chat

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"


def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="本田汽车销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")


def parse_arguments():
    argument_parser = ArgumentParser()
    return argument_parser.parse_arguments()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    print(args)
    # 初始化本田汽车销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
