llm = ChatOpenAI(
    model="your_model",
    temperature=0,
    max_tokens=1024,
    timeout=30,
    max_retries=3,
    api_key="your_api_key",
    base_url="your_base_url",
)
system_prompt = (
    "你是一个智能的导诊助手，你需要根据提供的信息来回答出合适的科室。"
    "\n\n"
    "请严格按照如下要求进行输出"
    "\n\n"
    "{query}"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", "{query}")])
correct_predictions = 0
total_predictions = 0
y_true = []  
y_pred = []  
jsonl_file_path = "your_testing_set_path" 
with open(jsonl_file_path, 'r') as file:
    for line in tqdm(file, desc="Processing", unit="line"):
        data = json.loads(line)
        query = data.get("question")  
        answer = data.get("answer")  
        if query and answer:
            res = llm.invoke(prompt.format(query=query))  
            model_response = res.content.strip()  
            model_response_cleaned = " ".join(model_response.split())  
            answer_cleaned = " ".join(answer.split())  
            y_true.append(answer_cleaned)
            y_pred.append(model_response_cleaned)
            if model_response_cleaned == answer_cleaned:
                correct_predictions += 1  
            total_predictions += 1  
TP, FP, FN = 0, 0, 0
unique_classes = set(y_true + y_pred)
for cls in unique_classes:
    for true, pred in zip(y_true, y_pred):
        true_set = set(true.split("|"))
        pred_set = set(pred.split("|"))
        TP += len(true_set & pred_set)  
        FP += len(pred_set - true_set)  
        FN += len(true_set - pred_set)  
precision = TP / (TP + FP) if TP + FP > 0 else 0
recall = TP / (TP + FN) if TP + FN > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")