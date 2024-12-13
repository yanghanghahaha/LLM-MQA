from data_utils import MyDataset
from api_utils import api_handler
from string import punctuation
import argparse
import tqdm
import json
from utils import *
from datetime import datetime
from score import acc_compute
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='chatgpt')
    parser.add_argument('--dataset_name', default='MedQA')
    parser.add_argument('--dataset_dir', default='./datasets/MedQA/')
    parser.add_argument('--start_pos', type=int, default=21)
    parser.add_argument('--end_pos', type=int, default=50)
    parser.add_argument('--output_files_folder', default='./outputs/MedQA')

    parser.add_argument('--method', type=str, default='base_cot_sc_fewshot', choices=['base_cot_sc_fewshot','base_few_shot_cot','base_few_shot_direct','syn_verif', 'syn_only', 'anal_only', 'base_direct', 'base_cot', 'base_cot+SC'])
    parser.add_argument('--max_attempt_vote', type=int, default=3)
    args = parser.parse_args()

    print(args)

    ### get handler
    if args.model_name in ['instructgpt', 'newinstructgpt', 'chatgpt', 'gpt4', 'ollama']: # select the model
        handler = api_handler(args.model_name)
    else:
        try :
            handler = api_handler(args.model_name)
            if ':' in args.model_name:
                args.model_name = args.model_name.replace(":", "_")
        except:
            raise ValueError

    ### get dataobj
    # dataobj = MyDataset('test', args, traindata_obj=None)
    dataobj = MyDataset('test', args, traindata_obj=None)
    ### set test range
    end_pos = len(dataobj) if args.end_pos == -1 else args.end_pos
    test_range = range(args.start_pos, end_pos)  # closed interval
    #
    ### set output_file_name
    output_time = datetime.now().strftime("%m%d_%H%M")
    exact_output_file = f"{args.output_files_folder}/{args.model_name}-{args.method}-{output_time}"
    print(exact_output_file)
    fail = {
        'fail_clean': [],
        'fail_try': [],
    }# 记录没通过clean的和try后仍然没过clean的
    input_prompt = {}

    start_time = time.time()
    # 这行代码使用了 tqdm 库中的 tqdm 函数，用于在循环中显示进度条
    for idx in tqdm.tqdm(test_range, desc=f"{args.start_pos} ~ {end_pos}"):
        raw_sample = dataobj.get_by_idx(idx)
        # print(raw_sample)
        # idx问题数的索引
        # options 选项
        # gold_answer 选项答案
        # question 是数据集里的问题
        question = raw_sample['question'] if raw_sample['question'][-1] in punctuation else raw_sample['question'] + '?'
        # print(question)
        realqid = idx
        # print(idx)
        if args.dataset_name in ['MedQA', 'MedMCQA'] or 'MMLU' in args.dataset_name:
            options = raw_sample['options']
            # print(options)
            gold_answer = raw_sample['answer_idx']
            # print(gold_answer)
        elif args.dataset_name == 'PubMedQA':
            question = raw_sample['context'] + ' ' + question
            options = raw_sample['options']
            gold_answer = raw_sample['answer_idx']

        elif args.dataset_name in ['MedicationQA']:
            options = ''
            gold_answer = raw_sample['answer_idx']

        data_info, fail_clean, fail_try = fully_decode(idx, realqid, question, options, gold_answer, handler, args, dataobj)

        if fail_clean: fail['fail_clean'].append((realqid, question[:50]))
        if fail_try: fail['fail_try'].append((realqid, question[:50]))

        record = json.dumps(data_info)
        # 是 Python 中 JSON 模块的一个函数，用于将 Python 对象（如字典、列表等）转换为 JSON 格式的字符串。

        if not os.path.exists(args.output_files_folder):
            os.makedirs(args.output_files_folder)
        with open(f'{exact_output_file}.jsonl', 'a') as f:
            f.write(record + '\n')
    end_time = time.time()
    time_taken = end_time - start_time
    acc_info = acc_compute(f'{exact_output_file}.jsonl')
    with open(f'{exact_output_file}.log', 'w') as f:
        fail_record = json.dumps(fail)
        acc_record = json.dumps(acc_info)
        f.write(str(args))
        t = f'[Time] {time_taken:.2f} seconds, average {time_taken/len(test_range):.2f} / question'
        f.write(t)
        f.write(fail_record)
        f.write(acc_record)