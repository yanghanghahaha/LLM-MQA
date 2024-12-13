import json
from random import random

from prompt_generator import *
from data_utils import *

def load_few_shot_examples(filepath, num_examples=3):
    """
    从 JSONL 文件中加载 few-shot 示例
    :param filepath: 数据集的文件路径
    :param num_examples: 需要的 few-shot 示例数量
    :return: 包含 few-shot 示例的列表
    """
    examples = []

    # 逐行读取 jsonl 文件
    with open(filepath, 'r') as file:
        for line in file:
            try:
                example = json.loads(line.strip())  # 确保解析为字典
                if isinstance(example, dict):  # 检查是否是字典类型
                    examples.append({
                        "question": example['question'],
                        "options": example['options'],
                        "answer": example['answer_idx']
                    })
                else:
                    print(f"Warning: Skipping malformed example: {line}")
            except Exception as e:
                print(f"Error parsing line: {line}\n{e}")

    # 随机选择指定数量的 few-shot 示例
    if len(examples) > num_examples:
        return random.sample(examples, num_examples)
    return examples

def fully_decode(qid, realqid, question, options, gold_answer, handler, args, dataobj,):
    fail_clean, fail_try = False, False
    # method = syn_verif
    question_domains, options_domains, question_analyses, option_analyses, syn_report, output = "", "", "", "", "", ""
    vote_history, revision_history, syn_repo_history = [], [], []
    case = []
    few_shot_examples = load_few_shot_examples('datasets/MedQA/few_shot.jsonl' , num_examples=3)
    print(few_shot_examples)
    if args.method == "base_direct":
        direct_prompt = get_direct_prompt(question, options)
        output = handler.get_output_multiagent(user_input=direct_prompt, temperature=0, max_tokens=50, system_role="")
        ans, output = cleansing_final_output(output)
    elif args.method == "base_cot":
        cot_prompt = get_cot_prompt(question, options)
        output = handler.get_output_multiagent(user_input=cot_prompt, temperature=0, max_tokens=300, system_role="")
        ans, output = cleansing_final_output(output)
    elif args.method == "base_cot+SC":
        cot_prompt = get_cot_sc_prompt(question, options)
        print(cot_prompt)
        output = handler.get_output_multiagent(user_input=cot_prompt, temperature=0, max_tokens=300, system_role="")
        ans, output = cleansing_final_output(output)
    elif args.method == "base_few_shot_direct":
        direct_prompt = get_few_shot_direct_prompt(question, options, few_shot_examples)
        print(direct_prompt)
        output = handler.get_output_multiagent(user_input=direct_prompt, temperature=0, max_tokens=50, system_role="")
        ans, output = cleansing_final_output(output)
    elif args.method == "base_few_shot_cot":
        cot_prompt = get_cot_prompt_with_fewshot(question, options, few_shot_examples)
        output = handler.get_output_multiagent(user_input=cot_prompt, temperature=0, max_tokens=300, system_role="")
        ans, output = cleansing_final_output(output)
    elif args.method == "base_cot_sc_fewshot":
        cot_prompt = get_cot_sc_fewshot_prompt(question, options, few_shot_examples)
        print(cot_prompt)
        output = handler.get_output_multiagent(user_input=cot_prompt, temperature=0, max_tokens=300, system_role="")
        ans, output = cleansing_final_output(output)
    else:
        # get question domains
        question_classifier, prompt_get_question_domain = get_question_domains_prompt(question)
        # question_classifier 是向模型提出他是一个专家的问题
        # prompt_get_question_domain 是向模型提出相关问题
        print(prompt_get_question_domain)
        raw_question_domain = handler.get_output_multiagent(user_input=prompt_get_question_domain, temperature=0,
                                                            max_tokens=50, system_role=question_classifier)
        # max_tokens=30
        print(raw_question_domain)
        example_classifier, prompt_get_example_domain = get_example_domains_prompt(question, options)
        raw_example_domain = handler.get_output_multiagent(user_input=prompt_get_example_domain, temperature=0,
                                                           max_tokens=500, system_role=question_classifier)
        print(raw_example_domain)
        case = [raw_example_domain]
        # raw_question_domain 是ai给出的相关答案
        if raw_question_domain == "ERROR.":
            raw_question_domain = "Medical Field: " + " | ".join(["General Medicine" for _ in range(NUM_QD)])
        question_domains = raw_question_domain.split(":")[-1].strip().split(" | ")
        print(question_domains)
        # question_domains :['Surgery', 'Orthopaedics', 'Medical Ethics', 'Patient Communication', 'Professional Responsibility']
        # get option domains
        options_classifier, prompt_get_options_domain = get_options_domains_prompt(question, options)
        print(prompt_get_options_domain)
        raw_option_domain = handler.get_output_multiagent(user_input=prompt_get_options_domain, temperature=0,
                                                          max_tokens=50, system_role=options_classifier)

        # max_tokens=50
        print(raw_option_domain)

        if raw_option_domain == "ERROR.":
            raw_option_domain = "Medical Field: " + " | ".join(["General Medicine" for _ in range(NUM_OD)])
        options_domains = raw_option_domain.split(":")[-1].strip().split(" | ")
        # **Medical Field: Ethics | Professionalism**
        # get question analysis
        tmp_question_analysis = []
        for _domain in question_domains:
            # print('11',_domain)
            # print('22',question_domains)
            question_analyzer, prompt_get_question_analysis = get_question_analysis_prompt(question, _domain)
            raw_question_analysis = handler.get_output_multiagent(user_input=prompt_get_question_analysis,
                                                                  temperature=0, max_tokens=500,
                                                                  system_role=question_analyzer)
            tmp_question_analysis.append(raw_question_analysis)
        # print(tmp_question_analysis)
        question_analyses = cleansing_analysis(tmp_question_analysis, question_domains, 'question')

        # tmp_question_analysis是模型给出的问题分析的集合
        # get option analysis
        tmp_option_analysis = []
        for _domain in options_domains:
            option_analyzer, prompt_get_options_analyses = get_options_analysis_prompt(question, options, _domain,
                                                                                       question_analyses)
            raw_option_analysis = handler.get_output_multiagent(user_input=prompt_get_options_analyses, temperature=0,
                                                                max_tokens=500, system_role=option_analyzer)
            tmp_option_analysis.append(raw_option_analysis)
        option_analyses = cleansing_analysis(tmp_option_analysis, options_domains, 'option')
        # print('aaa',option_analyses)
        if args.method == "anal_only":
            answer_prompt = get_final_answer_prompt_analonly(question, options, question_analyses, option_analyses)
            output = handler.get_output_multiagent(user_input=answer_prompt, temperature=0, max_tokens=2500,
                                                   system_role="")
            ans, output = cleansing_final_output(output)
        else:
            # get synthesized report
            q_analyses_text = transform_dict2text(question_analyses, "question", question)
            o_analyses_text = transform_dict2text(option_analyses, "options", options)
            print(q_analyses_text)
            print('---------')
            print(o_analyses_text)
            synthesizer, prompt_get_synthesized_report = get_synthesized_report_prompt(q_analyses_text, o_analyses_text,
                                                                                       case)

            # TODO: 即使使用qwen:72b，仍有一些输出不能通过clean
            # 受到影响的是syn_report, ans, output
            # 尝试解决方法，temputure=0.2赋予随机性，然后重复，直至达到输出标准
            # 测试案例：qwen:72b 的Q11: A 72-year-
            try:
                raw_synthesized_report = handler.get_output_multiagent(user_input=prompt_get_synthesized_report,
                                                                       temperature=0, max_tokens=2500,
                                                                       system_role=synthesizer)
                # print('-----------------------------')
                # print(raw_synthesized_report)
                syn_report = cleansing_syn_report(question, options, raw_synthesized_report)
                print(syn_report)
            # IndexError 异常通常在尝试访问序列（如列表、元组或字符串）中不存在的索引时引发。索引是用来指定序列中特定元素位置的整数值.
            # 当试图使用超出序列范围的索引时，Python 将引发 IndexError 异常。
            except IndexError as e:
                fail_clean = True
                attempt = 1
                while attempt <= 10:
                    print(f'[Try] reclean {attempt = }')
                    try:
                        raw_synthesized_report = handler.get_output_multiagent(user_input=prompt_get_synthesized_report,
                                                                               temperature=0, max_tokens=2500,
                                                                               system_role=synthesizer)
                        syn_report = cleansing_syn_report(question, options, raw_synthesized_report)
                        break
                    except IndexError as e:
                        attempt += 1
                else:  # 5次尝试都失败，无法生成syn_report返回空值
                    fail_try = True
                    print(f"[Failure] Q{qid}: {question[:10]} fail to clean synthesized report")
                    ans, output = "", ""
                    data_info = {
                        'question': question,
                        'options': options,
                        'pred_answer': ans,
                        'gold_answer': gold_answer,
                        'question_domains': question_domains,
                        'option_domains': options_domains,
                        'case': case,
                        'question_analyses': question_analyses,
                        'option_analyses': option_analyses,
                        'syn_report': syn_report,
                        'vote_history': vote_history,
                        'revision_history': revision_history,
                        'syn_repo_history': syn_repo_history,
                        'raw_output': output
                    }
                    return data_info, fail_clean, fail_try

            if args.method == "syn_only":
                # final answer derivation
                answer_prompt = get_final_answer_prompt_wsyn(syn_report)
                output = handler.get_output_multiagent(user_input=answer_prompt, temperature=0, max_tokens=2500,
                                                       system_role="")
                ans, output = cleansing_final_output(output)
            elif args.method == "syn_verif":
                all_domains = question_domains + options_domains
                # all_options = options_domains
                syn_repo_history = [syn_report]

                hasno_flag = True  # default value: in order to get into the while loop
                num_try = 0

                while num_try < args.max_attempt_vote and hasno_flag:
                    domain_opinions = {}  # 'domain' : 'yes' / 'no'
                    revision_advice = {}
                    num_try += 1
                    hasno_flag = False
                    # hold a meeting for all domain experts to vote and gather advice if they do not agree
                    # for domain in all_domains:
                    for domain in all_domains:
                        print(domain)
                        voter, cons_prompt = get_consensus_prompt(domain, syn_report)
                        raw_domain_opi = handler.get_output_multiagent(user_input=cons_prompt, temperature=0,
                                                                       max_tokens=30, system_role=voter)
                        domain_opinion = cleansing_voting(raw_domain_opi)  # "yes" / "no"
                        domain_opinions[domain] = domain_opinion
                        if domain_opinion == "no":
                            advice_prompt = get_consensus_opinion_prompt(domain, syn_report)
                            advice_output = handler.get_output_multiagent(user_input=advice_prompt, temperature=0,
                                                                          max_tokens=500, system_role=voter)
                            revision_advice[domain] = advice_output
                            hasno_flag = True
                    if hasno_flag:
                        revision_prompt = get_revision_prompt(syn_report, revision_advice)
                        revised_analysis = handler.get_output_multiagent(user_input=revision_prompt, temperature=0.95,
                                                                         max_tokens=2500, system_role="")
                        syn_report = cleansing_syn_report(question, options, revised_analysis)
                        revision_history.append(revision_advice)
                        syn_repo_history.append(syn_report)
                    vote_history.append(domain_opinions)
                print(syn_report)
                # final answer derivation
                answer_prompt = get_final_answer_prompt_wsyn(syn_report)
                output = handler.get_output_multiagent(user_input=answer_prompt, temperature=0, max_tokens=50,
                                                       system_role="")
                # forget_prompt = "Please remember to forget all previous conversations after selecting the best answer."
                # handler.get_output_multiagent(user_input=forget_prompt)
                # print(output)
                ans, output = cleansing_final_output(output)

    data_info = {
        'question': question,
        'options': options,
        'pred_answer': ans,
        'gold_answer': gold_answer,
        'question_domains': question_domains,
        'option_domains': options_domains,
        'case': case,
        'question_analyses': question_analyses,
        'option_analyses': option_analyses,
        'syn_report': syn_report,
        'vote_history': vote_history,
        'revision_history': revision_history,
        'syn_repo_history': syn_repo_history,
        'raw_output': output
    }

    return data_info, fail_clean, fail_try