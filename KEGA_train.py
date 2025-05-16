import time
import json
import os
from uuid import uuid4
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

start = time.time()
llm = ChatOpenAI(
    model="your_model",
    temperature=0,
    max_tokens=1024,
    timeout=30,
    max_retries=3,
    api_key="your_api_key",
    base_url="your_base_url",
)
model_name = "your_embedding_model_path"
device = 'cuda:0'
model_kwargs = {'device': device}
hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
embeddings = hf
if not os.path.exists("./MEK_Base"):
    txt_directory = "Medical_Literature_and_Records_path"
    txt_files = os.listdir(txt_directory)
    txt_file_list = []
    for entry in txt_files:
        full_path = os.path.join(txt_directory, entry)
        if os.path.isfile(full_path):
            txt_file_list.append(full_path)
    texts = []
    for file_path in txt_file_list:
        loader = TextLoader(file_path, encoding='utf8')
        text = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=512, 
                        chunk_overlap=200,
                        length_function=len,
                        is_separator_regex=False,
                        )
        texts += text_splitter.split_documents(text)
    vectorstore3 = Chroma.from_documents(
        collection_name="MEK_Base",
        documents=texts, 
        embedding=embeddings,
        persist_directory="./MEK_Base"
    )
else :
    vectorstore3 = Chroma(
        collection_name="MEK_Base",
        embedding_function=embeddings,
        persist_directory="./MEK_Base"
    )
vectorstore1 = Chroma(collection_name="EB", embedding_function=embeddings, persist_directory="./EB")
vectorstore2 = Chroma(collection_name="RB", embedding_function=embeddings, persist_directory="./RB")
sensitive_words = []
sensitive_words_directory = "your_data_path" 
for filename in os.listdir(sensitive_words_directory):
    file_path = os.path.join(sensitive_words_directory, filename)
    if os.path.isfile(file_path) and file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            sensitive_words.extend(f.read().splitlines())  
def contains_sensitive_word(query):
    return any(word in query for word in sensitive_words)
system_prompt = (
    "你是一个智能的导诊助手，你需要根据提供的信息来回答出合适的科室。"
    "\n\n"
    "请严格按照如下要求进行输出"
    "要求如下：\n"
    "1. 输出只能包含候选科室列表中的表项，不要赘述！\n"
    "2. 答案可能包含一个或多个科室，科室之间用\"|\"分隔。"
    "3. 请用中文作答"
    "\n\n"
    "这是从经验库中检索出来的相关内容为"
    "{context1}\n"
    "这是从反思库中检索出来的相关内容为"
    "{context2}\n"
    "这是从医学外部知识库中检索出来的相关内容为"
    "{context3}\n"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", "{query}")])
reflection1_system_prompt = (
    "之前的回答有误，现在请依据以下信息重新反思。\n"
    "之前回答的错误答案为"
    "{first_answer}\n"
    "该问题标准答案是"
    "{answer}\n"
    "请根据标准答案分析反思过程"
    "\n\n"
    "请按以下步骤进行反思，从对疾病症状理解错误、对科室职责认识不清等方面说明，写出详细的反思过程\n"
    "1. 分析错误原因，思考是否因对疾病症状、科室职责、问题关键信息理解偏差等导致回答错误，明确错误所在并详细说明\n"
    "2. 反思过程要结合所有检索内容，阐述如何依据标准答案修正思考方向，重新梳理逻辑\n"
    "3. 不要过多赘述，自己总结反思过程，语言精炼一些\n"
)
reflection2_system_prompt = (
    "已知反思过程为"
    "{reflection_process}\n"
    "之前的错误回答是"
    "{first_answer}\n"
    "该问题标准答案是"
    "{answer}\n"
    "请根据上述反思过程及相关信息进行再次作答。"
    "必须严格按照如下要求进行输出"
    "要求如下\n"
    "1. 输出只能包含候选科室列表中的表项，不要赘述！\n"
    "2. 答案可能包含一个或多个科室，科室之间用\"|\"分隔。"
    "\n\n"
)
reflection_prompt1 = ChatPromptTemplate.from_messages([("system", reflection2_system_prompt), ("user", "{query}")])
reflection_prompt2 = ChatPromptTemplate.from_messages([("system", reflection1_system_prompt), ("user", "{query}")])
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
rag_chain = (
    {"context1": vectorstore1.as_retriever(search_type="similarity", search_kwargs={"k": 3}) | format_docs,
     "context2": vectorstore2.as_retriever(search_type="similarity", search_kwargs={"k": 3}) | format_docs,
     "context3": vectorstore3.as_retriever(search_type="similarity", search_kwargs={"k": 1}) | format_docs,
     "query": RunnablePassthrough() }
    | prompt
    | llm
    | StrOutputParser()
)
jsonl_file_path = "your_training_set_path"
with open(jsonl_file_path, 'r') as file:
    lines = file.readlines()  
    total_lines = len(lines)  
    with tqdm(total=total_lines, desc="Processing Queries") as pbar:  
        for line in lines:
            data = json.loads(line)
            query = data.get("question")  
            answer = data.get("answer") 
            source = data.get("source")
            if query:
                if contains_sensitive_word(query):
                    continue  
                res = rag_chain.invoke(query)
                print("输入问题:", query)
                print("标准答案:", answer)
                print("模型初始响应:", res)
                if res == answer:
                    document = Document(
                        page_content=f"question: {query}\n answer: {res}\n source: {source}",
                        metadata={'row': len(vectorstore1.get()['ids'])}
                    )
                    vectorstore1.add_documents(documents=[document], ids=[str(uuid4())])
                    correct_predictions += 1 
                else:
                    reflection_chain1 = (
                        reflection_prompt2
                        | llm
                        | StrOutputParser()
                    )
                    context1_text = (vectorstore1.as_retriever(search_type="similarity", search_kwargs={"k": 3}) | format_docs).invoke(query)
                    context2_text = (vectorstore2.as_retriever(search_type="similarity", search_kwargs={"k": 3}) | format_docs).invoke(query)
                    context3_text = (vectorstore3.as_retriever(search_type="similarity", search_kwargs={"k": 1}) | format_docs).invoke(query)
                    reflection_process = reflection_chain1.invoke({
                        "context1": context1_text,
                        "context2": context2_text,
                        "context3": context3_text,
                        "query": RunnablePassthrough(),
                        "first_answer": res,
                        "answer": answer
                    })
                    reflection_chain2 = (
                        reflection_prompt1
                        | llm
                        | StrOutputParser()
                    )
                    context4_text = (vectorstore1.as_retriever(search_type="similarity", search_kwargs={"k": 3}) | format_docs).invoke(query)
                    context5_text = (vectorstore2.as_retriever(search_type="similarity", search_kwargs={"k": 3}) | format_docs).invoke(query)
                    context6_text = (vectorstore3.as_retriever(search_type="similarity", search_kwargs={"k": 1}) | format_docs).invoke(query)
                    reflection_res = reflection_chain2.invoke({
                        "context1": context4_text,
                        "context2": context5_text,
                        "context3": context6_text,
                        "query": RunnablePassthrough(),
                        "reflection_process": reflection_process,
                        "first_answer": res,
                        "answer": answer
                    })
                    if len(reflection_res) > 2048 or len(reflection_process) > 2048:
                        continue  
                    if reflection_res == answer:
                        document = Document(
                            page_content=( 
                                f"question: {query}\n"
                                f"first_answer: {res}\n"
                                f"reflection_process: {reflection_process}\n"
                                f"reflection_answer: {reflection_res}\n"
                                f"source: {source}"
                            ),
                            metadata={'row': len(vectorstore2.get()['ids'])}
                        )
                        vectorstore2.add_documents(documents=[document], ids=[str(uuid4())])
                        correct_predictions += 1  
                    else:
                        false_negatives += 1  
                if res != answer:
                    false_positives += 1  
                total_predictions += 1  