import json
from random import sample
# 假设MedQA数据集的JSONL文件名为 'medqa_dataset.jsonl'
file_name = 'datasets/MedQA/test.jsonl'
output_file_name = 'datasets/MedQA/test_parameter.jsonl'

# 读取JSONL文件并存储所有行
with open(file_name, 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

# 检查数据集的行数
if len(data) < 300:
    print("数据集的样本数少于300，请检查数据集文件。")
else:
    # 随机选取300个样本
    sample_data = sample(data, 5)

    # 将选取的样本保存为新的JSONL文件
    with open(output_file_name, 'w', encoding='utf-8') as output_file:
        for sample in sample_data:
            # 将JSON对象转换为字符串并写入文件
            json.dump(sample, output_file)
            output_file.write('\n')  # JSONL文件中每个JSON对象占一行，以换行符分隔