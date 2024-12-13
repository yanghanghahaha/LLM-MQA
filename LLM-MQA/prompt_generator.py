NUM_QD = 5
NUM_OD = 2


# question domains
def get_question_domains_prompt(question, max_words=50):
    question_domain_format = "Medical Field: " + " | ".join(["Field" + str(i) for i in range(NUM_QD)])
    question_classifier = ("You are a highly experienced medical expert specializing in classifying medical scenarios "
                           "into relevant areas of medicine. Your analysis should reflect the latest clinical guidelines "
                           "and best practices.")
    prompt_get_question_domain = (f"You need to complete the following steps:\n"
                                  f"1. Carefully read the medical scenario presented in the question: '''{question}'''.\n"
                                  f"2. Based on the medical scenario, classify the question into one distinct subfields of medicine. "
                                  f"Consider all potential differential diagnoses and related fields.\n"
                                  f"You should output in exactly the same format as '''{question_domain_format}."
                                  f"Keep your response concise and under {max_words} words.")
    return question_classifier, prompt_get_question_domain

def get_example_domains_prompt(question, options, T_words=200, max_words=500):
    """
    生成支持正确选项的案例，避免过度引导，同时确保案例内容客观和中立。
    """
    case_generator = (
        f"You are a medical expert. Your task is to generate concise and realistic cases "
        f"that support the analysis of the most plausible option. "
        f"These cases should provide clinical context relevant to the options, focusing on the diagnostic features or treatments that support the correct option."
    )

    prompt = (
        f"**Inputs**:\n"
        f"1. Question: '''{question}'''\n"
        f"2. Options: {options}\n\n"
        f"**Task**:\n"
        f"1. Carefully analyze the question and options provided.\n"
        f"2. Identify the most plausible correct option.\n"
        f"3. Create 1-2 realistic cases that:\n"
        f"   - Highlight the clinical reasoning behind supporting the correct option.\n"
        f"   - Provide a detailed clinical context, focusing on diagnostic findings, symptoms, or treatments relevant to the selected option.\n"
        f"   - While emphasizing the benefits of the correct option, avoid exaggerating its advantages, and provide a balanced view.\n\n"
        f"**Format for each case:**\n"
        "Case X:\n"
        "- Context: [Describe the clinical scenario, including symptoms, diagnostic findings, or treatments. Ensure the scenario reflects the question's clinical context.]\n"
        "- Key Mechanism/Reasoning: [Explain how this case supports the selected option, focusing on its clinical applicability. Avoid overstating its importance.]\n"
        "- Neutrality Check: [Ensure the case is unbiased and does not excessively favor the selected option. Be mindful of presenting a balanced view.]\n\n"
        f"- Keep each case concise, under {T_words} words.\n"
        f"- Ensure the total response remains under {max_words} words."
    )

    return case_generator, prompt


def get_question_analysis_prompt(question, question_domain, max_words=300):
    """
    优化问题分析模块，避免对选项分析模块产生诱导作用。
    """
    question_analyzer = (
        f"You are a medical expert in the domain of {question_domain}. "
        f"Your role is to carefully analyze the medical scenario presented in the question."
        f" Focus on interpreting the symptoms, history, and key findings, without suggesting specific diagnoses or next steps."
    )

    prompt_get_question_analysis = (
        f"Please carefully evaluate the following medical scenario: '''{question}'''.\n"
        f"Your task is to interpret the condition being depicted based on the provided clinical information. "
        f"Identify the critical symptoms, historical factors, and diagnostic clues that are most significant for understanding the case, "
        f"without suggesting specific treatment options or next steps. Keep your response concise and under {max_words} words."
    )

    return question_analyzer, prompt_get_question_analysis




def get_options_domains_prompt(question, options):
    options_domain_format = "Medical Fields: " + " | ".join(["Field" + str(i) for i in range(NUM_OD)])
    options_classifier = f"As a medical expert, you need to identify the two most relevant medical domains for addressing the question and options."

    prompt_get_options_domain = f"You need to complete the following steps:" \
                f"1. Carefully read the medical scenario presented in the question: '''{question}'''." \
                f"2. The available options are: '''{options}'''. Strive to understand the fundamental connections between the question and the options." \
                f"3. Your core aim should be to categorize the options into two distinct subfields of medicine. " \
                f"You should output in exactly the same format as '''{options_domain_format}'''" \
                # f"Keep your response concise and under {max_words} words."

    return options_classifier, prompt_get_options_domain

def get_options_analysis_prompt(question, options, op_domain, question_analysis, max_words=500):
    """
    优化选项分析模块，确保选项分析专注于独立评估每个选项的合理性。
    """
    # 描述选项分析专家的角色
    option_analyzer = (
        f"You are a medical expert specialized in the {op_domain} domain. "
        f"Your task is to analyze the available options for the given medical scenario. "
        f"Review each option independently, considering its clinical relevance, the patient’s condition, and established medical guidelines."
    )

    # 问题分析输出，确保它仅供参考，不影响选项分析
    prompt_get_options_analyses = f"Regarding the question: '''{question}''', the following are the options available: '''{options}'''.\n"
    prompt_get_options_analyses += f"You have been provided with expert analyses of the medical scenario from various domains, such as:\n"

    # 问题分析中提供的关键信息，简要提及，而不详细展开，避免引导分析
    for _domain, _analysis in question_analysis.items():
        prompt_get_options_analyses += f"From the {_domain} expert: {_analysis}.\n"

    prompt_get_options_analyses += (
        f"Your task is to independently evaluate each option from the perspective of your medical domain. "
        f"Examine the plausibility of each option in relation to the patient's clinical condition, diagnostic clues, and the evidence available. "
        f"Assess whether each option is the most appropriate next step or should be eliminated, based on medical reasoning and logic.\n\n"
        f"**Instructions**:\n"
        f"1. Analyze each option individually, considering supporting and refuting evidence for each.\n"
        f"2. Pay attention to subtle differences between the options and reason through their plausibility.\n"
        f"3. Ensure that your analysis is objective, focusing solely on the medical and clinical validity of each option, without drawing conclusions from the problem analysis provided.\n"
        f"4. Provide a clear justification for each option's plausibility or elimination.\n\n"
        f"**Output Format**:\n"
        f"Option [X]:\n"
        f"- **Plausibility Analysis**: [Provide your reasoning and evidence for why this option is plausible or not.]\n"
        f"- **Conclusion**: [Summarize your reasoning for keeping or eliminating this option.]\n"
        f"Ensure the response is under {max_words} words."
    )

    return option_analyzer, prompt_get_options_analyses

def get_final_answer_prompt_analonly(question, options, question_analyses, option_analyses):
    prompt = f"Question: {question} \nOptions: {options} \n" \
             f"Answer: Let's work this out in a step by step way to be sure we have the right answer. \n" \
             f"Step 1: Decode the question properly. We have a team of experts who have done a detailed analysis of this question. " \
             f"The team includes five experts from different medical domains related to the problem. \n"

    for _domain, _analysis in question_analyses.items():
        prompt += f"Insight from an expert in {_domain} suggests, {_analysis} \n"

    prompt += f"Step 2: Evaluate each presented option individually, based on both the specifics of the patient's scenario as well as your medical knowledge. " \
              f"Pay close attention to discerning the disparities among the different options. " \
              f"A handful of these options might seem right on the first glance but could potentially be misleading in reality. " \
              f"We have detailed analyses from experts across two domains. \n"

    for _domain, _analysis in option_analyses.items():
        prompt += f"Assessment from an expert in {_domain} suggests, {_analysis} \n"
    prompt += f"Step 3: Based on the understanding gathered from the above steps, select the optimal choice to answer the question. \n" \
              f"Points to note: \n" \
              f"1. The analyses provided should guide you towards the correct response. \n" \
              f"2. Any option containing incorrect information inherently cannot be the correct choice. \n" \
              f"3. Please respond only with the selected option's letter, like A, B, C, D, or E, using the following format: '''Option: [Selected Option's Letter]'''. " \
              f"Remember, it's the letter we need, not the full content of the option."

    return prompt

def get_final_answer_prompt_wsyn(syn_report):
    prompt = (f"Here is the synthesized report: {syn_report}\n"
              f"Steps:\n"
              f"1. Carefully review the synthesized report to identify the most plausible option (A, B, C, D, or E) based on the analysis. "
              f"If one option is clearly supported by the evidence, select it.\n"
              f"2. If no option is explicitly confirmed, evaluate each option by considering the following criteria:\n"
              f"   - Does the option align with the major findings in the report?\n"
              f"   - Does the option fit with the patient’s clinical context, including key symptoms, diagnostic results, and underlying conditions?\n"
              f"   - Does the option align with general medical reasoning and best practices?\n"
              f"3. If multiple options are plausible, consider the following steps:\n"
              f"   - Eliminate any options that are less supported or contradicted by the report.\n"
              f"   - Prioritize the option most consistent with the overall clinical reasoning and key evidence.\n"
              f"   - Ensure that the selected option is the one most likely to lead to the most accurate diagnosis or treatment based on the synthesized analysis.\n"
              f"4. Provide the selected option’s letter (A, B, C, D, or E) based on the above analysis. Respond only with the following format: '''Option: [Selected Option's Letter]'''."
              f"Please remember to forget all previous conversations after selecting the best answer.")
    return prompt



def get_direct_prompt(question, options):
    prompt = f"Question: {question} \n" \
             f"Options: {options} \n" \
             f"Please respond only with the selected option's letter, like A, B, C, D, or E, using the following format: '''Option: [Selected Option's Letter]'''."
    return prompt
def get_few_shot_direct_prompt(question, options, few_shot_examples):
    prompt = "Here are some examples of medical questions and answers:\n\n"

    # Add few-shot examples
    for example in few_shot_examples:
        prompt += f"Question: {example['question']}\n"
        prompt += f"Options: {example['options']}\n"
        prompt += f"Answer: {example['answer']}\n\n"
    prompt = f"Question: {question} \n" \
             f"Options: {options} \n" \
             f"Please respond only with the selected option's letter, like A, B, C, D, or E, using the following format: '''Option: [Selected Option's Letter]'''."
    return prompt

def get_cot_prompt(question, options, max_words=100):
    prompt = f"Question: {question} \n" \
             f"Options: {options} \n" \
             f"Let's carefully think step by step, analyzing each option in detail and reasoning through them logically. \n" \
             f"Keep your response concise and under {max_words} words. \n" \
             f"Please respond only with the selected option's letter, like A, B, C, D, or E, using the following format: '''Option: [Selected Option's Letter]'''."
    return prompt


def get_cot_prompt_with_fewshot(question, options, few_shot_examples, max_words=100):
    """
    在 COT 方法中加入 few-shot 示例，显式要求推理过程，生成包含推理过程的 prompt。

    :param question: 当前问题
    :param options: 当前问题的选项
    :param few_shot_examples: few-shot 示例列表
    :param max_words: 最大回答字数限制
    :return: 构建的 COT prompt 字符串
    """
    # 初始化 prompt
    prompt = "Here are some examples of reasoning through similar questions:\n\n"

    # 添加 few-shot 示例
    for example in few_shot_examples:
        prompt += f"Question: {example['question']}\n"
        prompt += f"Options: {example['options']}\n"
        prompt += f"Let's think step by step...\n"  # 添加推理提示
        prompt += f"Answer: {example['answer']}\n\n"

    # 添加当前问题
    prompt += f"Question: {question}\n"
    prompt += f"Options: {options}\n"
    prompt += f"Let's carefully think step by step, analyzing each option in detail and reasoning through them logically.\n"
    prompt += f"Keep your response concise and under {max_words} words.\n"
    prompt += f"Please respond only with the selected option's letter, like A, B, C, D, or E, using the following format: '''Option: [Selected Option's Letter]'''."

    return prompt

def get_cot_sc_prompt(question, options, max_words=100, num_resamples=3):
    prompt = f"Question: {question} \n" \
             f"Options: {options} \n" \
             f"Let's carefully think step by step, analyzing each option in detail and reasoning through them logically. \n" \
             f"### Step 1: Reasoning for Each Option (Multiple Resamples): \n" \
             f"Let's now generate multiple independent reasoning steps for each option to evaluate consistency. We will generate {num_resamples} different reasoning steps for each option: \n\n"

    for i in range(1, 6):
        prompt += f"#### Option {chr(64 + i)}): \n"
        for j in range(1, num_resamples + 1):
            prompt += f"Resample {j}: Reasoning step for Option {chr(64 + i)}. \n"

    prompt += f"\n### Step 2: Self-Consistency Check: \n" \
              f"Now, evaluate the consistency of the different reasoning steps generated for each option. Which option has the most consistent reasoning across different resamples? \n" \
              f"Please respond only with the selected option's letter, like A, B, C, D, or E, using the following format: '''Option: [Selected Option's Letter]'''."
    return prompt

def get_cot_sc_fewshot_prompt(question, options, few_shot_examples, max_words=100, num_resamples=3):
    """
    将 COT+SC 方法与 few-shot 示例结合，生成包含推理步骤和一致性评估的 prompt。

    :param question: 当前问题
    :param options: 当前问题的选项
    :param few_shot_examples: few-shot 示例列表
    :param max_words: 最大字数限制
    :param num_resamples: 每个选项生成的推理步骤数量
    :return: 构建的 COT+SC + few-shot prompt 字符串
    """
    # 初始化 prompt
    prompt = "Here are some examples of reasoning through similar questions:\n\n"

    # 添加 few-shot 示例
    for example in few_shot_examples:
        prompt += f"Question: {example['question']}\n"
        prompt += f"Options: {example['options']}\n"
        prompt += f"Let's carefully think step by step...\n"  # 强调推理过程
        prompt += f"### Step 1: Reasoning for Each Option (Multiple Resamples): \n"

        # 生成多个推理步骤
        for j in range(1, num_resamples + 1):
            prompt += f"Resample {j}: Reasoning step for each option. \n"

        prompt += f"### Step 2: Self-Consistency Check: \n" \
                  f"Now, evaluate the consistency of the reasoning steps for each option. Which option has the most consistent reasoning? \n"
        prompt += f"Answer: {example['answer']}\n\n"

    # 添加当前问题
    prompt += f"Question: {question}\n"
    prompt += f"Options: {options}\n"
    prompt += f"Let's carefully think step by step, analyzing each option in detail and reasoning through them logically.\n"
    prompt += f"### Step 1: Reasoning for Each Option (Multiple Resamples): \n"
    prompt += f"Let's now generate {num_resamples} different reasoning steps for each option to evaluate consistency: \n\n"

    for i in range(1, 6):
        prompt += f"#### Option {chr(64 + i)}): \n"
        for j in range(1, num_resamples + 1):
            prompt += f"Resample {j}: Reasoning step for Option {chr(64 + i)}. \n"

    prompt += f"\n### Step 2: Self-Consistency Check: \n" \
              f"Now, evaluate the consistency of the different reasoning steps generated for each option. Which option has the most consistent reasoning across different resamples? \n" \
              f"Please respond only with the selected option's letter, like A, B, C, D, or E, using the following format: '''Option: [Selected Option's Letter]'''."

    return prompt


def get_synthesized_report_prompt(question_analysis, option_analysis, case_analysis, max_words=500):
    """
    生成符合 Key Knowledge 和 Total Analysis 输出格式的综合报告 Prompt。
    """
    synthesizer = (
        "You are tasked with synthesizing a concise and comprehensive report based on multiple experts’ analyses. "
        "Your goal is to integrate insights from the Question Analysis, Option Analysis, and Case Analysis modules."
    )

    syn_report_format = (
        f"Key Knowledge: [Extract the critical diagnostic clues, clinical context, and reasoning from all analyses, "
        f"including the most important insights from the Question Analysis, Option Analysis, and Case Analysis.]\n"
        f"Total Analysis: [Synthesize the clinical scenario, evaluate each option based on supporting and refuting evidence, "
        f"and provide a ranked recommendation with a clear justification.]\n"
    )

    prompt_get_report = (
        f"**Inputs**:\n"
        f"1. Question Analysis: '''{question_analysis}'''\n"
        f"2. Option Analysis: '''{option_analysis}'''\n"
        f"3. Case Analysis: '''{case_analysis}'''\n\n"
        f"**Instructions**:\n"
        f"1. Summarize the clinical scenario from the Question Analysis, highlighting key diagnostic clues.\n"
        f"2. Extract Key Knowledge:\n"
        f"   - Include the main clinical context and critical diagnostic clues derived from the Question Analysis.\n"
        f"   - Highlight the most important supporting or refuting evidence for each option, derived from the Option Analysis.\n"
        f"   - Summarize the key insights provided by the Case Analysis.\n"
        f"3. Generate a Total Analysis:\n"
        f"   - Provide a concise summary of the clinical scenario.\n"
        f"   - For each option, integrate the supporting and refuting evidence from the Option Analysis and Case Analysis.\n"
        f"   - Rank the options by plausibility and explain the rationale for the final recommendation.\n\n"
        f"**Output Format**:\n{syn_report_format}\n"
        f"Ensure the report is concise and under {max_words} words."
    )
    return synthesizer, prompt_get_report





def get_consensus_prompt(domain, syn_report):
    voter = f"You are a medical expert specialized in the {domain} domain."
    cons_prompt = f"Here is a medical report: {syn_report} \n" \
                  f"As a medical expert specialized in {domain}, please carefully read the report and decide whether your opinions are consistent with this report." \
                  f"Please respond only with: [YES or NO]."
    return voter, cons_prompt


def get_consensus_opinion_prompt(domain, syn_report):
    opinion_prompt = f"Here is a medical report: {syn_report} \n" \
                     f"As a medical expert specialized in {domain}, please make full use of your expertise to propose revisions to this report." \
                     f"You should output in exactly the same format as '''Revisions: [proposed revision advice] '''"
    return opinion_prompt

def get_revision_prompt(syn_report, revision_advice):
    revision_prompt = f"Here is the original report: {syn_report}\n\n"
    for domain, advice in revision_advice.items():
        revision_prompt += f"Advice from a medical expert specializing in {domain}: {advice}.\n"
    revision_prompt += f"Based on the above advice, output the revised analysis in exactly the same format as '''Total Analysis: [revised analysis] '''"
    return revision_prompt