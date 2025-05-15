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
vectorstore1 = Chroma(collection_name="EB", embedding_function=embeddings, persist_directory="./EB")
vectorstore2 = Chroma(collection_name="RB", embedding_function=embeddings, persist_directory="./RB")
vectorstore3 = Chroma(collection_name="MEK_Base", embedding_function=embeddings, persist_directory="./MEK_Base")
system_prompt = (
    "你是一个智能的导诊助手，你需要根据提供的信息来回答出合适的科室。"
    "\n\n"
    "请严格按照如下要求进行输出"
    "要求如下：\n"
    "1. 输出只能包含候选科室列表中的表项，不要赘述！\n"
    "2. 答案可能包含一个或多个科室，科室之间用\"|\"分隔。"
    "\n\n"
    "这是从经验库中检索出来的相关内容为"
    "{context1}\n"
    "这是从反思库中检索出来的相关内容为"
    "{context2}\n"
    "这是从医学外部知识库中检索出来的相关内容为"
    "{context3}\n"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", "{query}")])
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
rag_chain = (
    {"context1": vectorstore1.as_retriever(search_type="similarity", search_kwargs={"k": 1}) | format_docs,  #k的值可更改
     "context2": vectorstore2.as_retriever(search_type="similarity", search_kwargs={"k": 1}) | format_docs,  #k的值可更改
     "context3": vectorstore3.as_retriever(search_type="similarity", search_kwargs={"k": 1}) | format_docs,  #k的值可更改
     "query": RunnablePassthrough() }
    | prompt
    | llm
    | StrOutputParser()
)
TP, FP, FN = 0, 0, 0
correct_predictions = 0  
total_predictions = 0  
y_true = []  
y_pred = []  
jsonl_file_path = "your_testing_set_path" 
with open(jsonl_file_path, 'r') as file:
    for line in tqdm(file, desc="Processing", unit="line"):
        line = line.strip()  
        if not line:  
            continue
        data = json.loads(line)  
        query = data.get("question")  
        answer = data.get("answer")  
        if query and answer:
            res = rag_chain.invoke(query)
            res_cleaned = " ".join(res.split())  
            answer_cleaned = " ".join(answer.split()) 
            y_true.append(answer_cleaned)
            y_pred.append(res_cleaned)
            true_set = set(answer_cleaned.split("|"))
            pred_set = set(res_cleaned.split("|"))
            TP += len(true_set & pred_set)  
            FP += len(pred_set - true_set)  
            FN += len(true_set - pred_set)  
            if res_cleaned == answer_cleaned:
                correct_predictions += 1
            total_predictions += 1
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")