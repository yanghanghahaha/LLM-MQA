import json

def acc_compute(file):
    acc = 0
    null = 0
    null_idx = []
    wrong_idx = []
    with open(file, 'r', encoding="utf-8") as f:
        total_num = sum(1 for _ in f)
        f.seek(0)# 重新定位文件指针到文件开头
        for idx, line in enumerate(f):
            j = json.loads(line)
            pred_answer = j['pred_answer']
            gold_answer = j['gold_answer']
            # pred_answer = j['pred_sentiment']
            # gold_answer = j['gold_sentiment']
            # print(f'{pred_answer=}{gold_answer=}{pred_answer == gold_answer}')
            if pred_answer == gold_answer:
                acc += 1
            elif pred_answer == "":
                null += 1
                null_idx.append(idx)
            else: 
                wrong_idx.append(idx)
        acc_info = {
            'acc': round(acc/total_num,4),
            'total_num': total_num,
            'acc_num': acc,
            'wrong_num':len(wrong_idx),
            'null_num':len(null_idx),
            'wrong_idx(jsonl)':wrong_idx,
            'null_idx(jsonl)':null_idx,
        }
    return acc_info
            
if __name__ == '__main__':
    file = 'medagents/outputs/llama3.1_70b-syn_verif-1118_2219.jsonl'
    acc_info = acc_compute(file)
    print(acc_info)
    with open(f'{file}.log', 'w') as f:
        acc_record = json.dumps(acc_info, indent=4)
        f.write(acc_record)

