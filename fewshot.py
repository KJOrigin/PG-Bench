import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

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
    "以下是一个例子，可供参考" # Examples can be replaced
    "\n\n"
    "{example_query}"
    "{example_answer}"
    "\n\n"
    "请严格按照如下要求进行输出"
    "\n\n"
    "{query}"
)
example_query = (
    "任务：请根据医患对话信息，从给定的候选科室中，为患者找到合适的挂号科室。\n候选科室列表如下：('预防保健科', '全科医疗科', '内科', '外科', '妇产科', '妇女保健科', '儿科', '小儿外科', '儿童保健科', '眼科', '耳鼻咽喉科', '口腔科', '皮肤科', '医疗美容科', '精神科', '传染科', '结核病科', '地方病科', '肿瘤科', '急诊医学科', '康复医学科', '运动医学科', '职业病科', '临终关怀科', '特种医学与军事医学科', '麻醉科', '疼痛科', '重症医学科', '医学检验科', '病理科', '医学影像科', '中医科', '中西医结合科', '其他业务科室')\n 要求：\n1. 输出只能包含候选科室列表中的表项，不要赘述；\n2. 答案可能包含一个或多个科室，科室之间用\"|\"分隔。\n\n医患对话信息如下：\n患者：医生，我今晨起后感觉颈肩部疼痛，活动受限，已经持续了半天了，没有明显的外伤史（女，37岁）。\n医生：疼痛的感觉是持续性的吗？还是有时加重，有时缓解？\n患者：是持续性的，感觉有点酸痛，尤其在活动时，疼痛会加重。\n医生：您有没有感到头晕、手臂麻木或者其他不适？\n患者：没有头晕，也没有手臂麻木，只是颈肩部疼痛和活动受限。\n推荐科室："
)
example_answer = ("外科|疼痛科")
formatted_system_prompt = system_prompt.format(
    example_query=example_query, 
    example_answer=example_answer,
    query="{query}"
)
prompt = ChatPromptTemplate.from_messages([("system", formatted_system_prompt), ("user", "{query}")])
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