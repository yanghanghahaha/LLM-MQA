import json

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from sklearn.metrics import classification_report

acc = 0
null = 0
wrong = 0
y_true = []
y_pred = []
y_revised = []
with open('outputs/few_shot_cot_sc.jsonl', 'r', encoding="utf-8") as f:
    total_num = sum(1 for _ in f)
    f.seek(0)  # 重新定位文件指针到文件开头
    for idx, line in enumerate(f):
        j = json.loads(line)
        y_true.append(j['gold_answer'])
        y_pred.append(j['pred_answer'])
        pred_answer = j['pred_answer']
        gold_answer = j['gold_answer']

        if pred_answer == gold_answer:
            acc += 1
        elif pred_answer == "":
            null += 1
        else:
            wrong += 1
            print(idx)
        # if j['revision_history'] and len(j['question'])<=350:
        #     print(idx)
        # print(f'{pred_answer=}{gold_answer=}{pred_answer == gold_answer}')
accuracy = accuracy_score(y_true, y_pred)
print(null)
print(wrong)
print(y_true)
print(y_pred)
assert len(y_true) == len(y_pred), "真实值和预测值长度不一致！"
# 统计 y_true 中每个类别的数量
category_counts = Counter(y_true)
category_counts1 = Counter(y_pred)
# 输出类别及其对应的样本数量
print(category_counts)
print(category_counts1)
for idx, value in enumerate(y_pred):
    if value == '':
        print(f"Empty string found at index {idx}")


print(f'Accuracy: {accuracy:.3f}')
precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
print(f'Macro Precision: {precision_macro:.3f}')
print(f'Macro Recall: {recall_macro:.3f}')
print(f'Macro F1 Score: {f1_macro:.3f}')
