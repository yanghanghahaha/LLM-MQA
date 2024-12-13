<<<<<<< HEAD
# LLM-MQA
=======
# 用ollama部署MedAgents

原README：[gersteinlab/MedAgents (github.com)](https://github.com/gersteinlab/MedAgents)，需要其它测试dataset的请回原README找

![](pics/overview.png)

## ollama模型下载部署

```
# linux 安装
curl -fsSL https://ollama.com/install.sh | sh
```

它会自动让ollama加入系统命令，之后可以直接调用ollama xxx

[ollama官网提供的模型](https://ollama.com/library)

这里以Qwen为例，沿用[Qwen1.5GIthub](https://github.com/QwenLM/Qwen1.5?tab=readme-ov-file#-run-locally)说明文档部署，先initiate the ollama service，生成private key

```
ollama serve
```

> Error: listen tcp 127.0.0.1:11434: bind: address already in use

默认是127.0.0.1:11434

下载并运行模型，以Qwen 1.5的0.5b模型为例

```
ollama run qwen:0.5b
# To exit, type "/bye" and press ENTER
```

>❯ ollama run qwen:0.5b
>pulling manifest 
>pulling fad2a06e4cc7... 100%                    
>pulling 41c2cf8c272f... 100%                                 
>verifying sha256 digest 
>writing manifest 
>removing any unused layers 
>success 

之后直接进入问答模式，说明ollama部署成功。输入`/bye`以退出

##  修改后的MedAgent代码使用说明

**省流版本：个人fork已修改完毕，可以直接运行**

测试环境：

- OS: Ubuntu 22.04.4 LTS on Windows 10 x86_64 
- Kernel: 5.15.146.1-microsoft-standard-WSL2 
- model: `qwen:0.5b` 

1. `inference.sh `中，`model_name`改成ollama模型名称。具体ollama模型名称请运行命令行

```
ollama list
```

2. 默认跑`MedQA`数据集需要安装如下分词

```
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

3. 最后，

```
cd MedAgents
sh inference.sh                                                                        
```

**报错警告！！！**

```
  File "/home/sakura/MedAgents/run.py", line 55, in <module>
    data_info = fully_decode(idx, realqid, question, options, gold_answer, handler, args, dataobj)
  File "/home/sakura/MedAgents/utils.py", line 60, in fully_decode
    syn_report = cleansing_syn_report(question, options, raw_synthesized_report)
  File "/home/sakura/MedAgents/data_utils.py", line 154, in cleansing_syn_report
    total_analysis_text = tmp[1].strip()
IndexError: list index out of range
```

直接原因：`cleansing_syn_report()`清洗LLM输出时无法依据给定关键词提取

根本原因：小模型参数性能差，答非所问。已尝试`qwen:0.5`,`qwen:1.8`效果都不行，都会报错

本代码修改中，已经在终端中print大模型输出结果，以便将来`cleansing_syn_report()`调试

# 附录

## 一些ollama操作

- 卸载ollama。见官方**linux部署**指引[ollama/docs/linux.md at main · ollama/ollama (github.com)](https://github.com/ollama/ollama/blob/main/docs/linux.md#uninstall)

- 模型默认下载的位置，见**官方QA**[ollama/docs/faq.md at main · ollama/ollama (github.com)](https://github.com/ollama/ollama/blob/main/docs/faq.md#where-are-models-stored)

```
Linux: /usr/share/ollama/.ollama/models
```

- 运行本地GGUF量化大模型

[Ollama 通过GGUF 文件本地运行任何开源大模型-CSDN博客](https://blog.csdn.net/qq_42881308/article/details/137110873#:~:text=Ollama 可以在本地运行任何开源大模型 只要下载到 GGUF 文件（相当于压缩的大模型） ** 1、下载 GGUF,txt 文建 3、打开ollama 终端输入：ollama create baichuan2-7b -f Modelflie.txt)

- 关闭ollama serve

目前只能杀进程

```
systemctl stop ollama.service # Ubuntun
sudo killall -s 9 ollama # WSL
```

开启ollama serve 时，nvidia-smi就会显示有进程占用GPU

## 源代码修改过程

参考文档

> ollama引导[OpenAI compatibility · Ollama Blog](https://ollama.com/blog/openai-compatibility)
>
> openai的api说明文档[openai/openai-python: The official Python library for the OpenAI API (github.com)](https://github.com/openai/openai-python?tab=readme-ov-file#documentation)

1. `requirements.txt`中，openai的版本应为1.14.3

```
openai==1.14.3
```

否则报错

>  cannot import name 'OpenAI' from 'openai'

2. 根据新openai的语法修改相应input和raise error

3. 可能会出现如下报错

> LookupError: 
>
> **********************************************************************
>
> Resource punkt not found.
> Please use the NLTK Downloader to obtain the resource:
>
>   >import nltk
>   >
>   >nltk.download('punkt')
>
> For more information see: https://www.nltk.org/data.html
>
> Attempted to load tokenizers/punkt/PY3/english.pickle

那就安装咯

```
python -c "import nltk; nltk.download('punkt')"
```

4. 报错指引：

    ```
    Traceback (most recent call last):
      File "/home/sakura/MedAgents/run.py", line 52, in <module>
        data_info = fully_decode(idx, realqid, question, options, gold_answer, handler, args, dataobj)
      File "/home/sakura/MedAgents/utils.py", line 60, in fully_decode
        syn_report = cleansing_syn_report(question, options, raw_synthesized_report)
      File "/home/sakura/MedAgents/data_utils.py", line 154, in cleansing_syn_report
        total_analysis_text = tmp[1].strip()
    IndexError: list index out of range
    ```

    关注`api_utils.py`中，`class api_handler`下，`get_output_multiagent()`函数中，`response`的值。

    参考[How to format inputs to ChatGPT models | OpenAI Cookbook](https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models)，正文输出应改为

    ```
    response.choices[0].message.content
    ```

>>>>>>> 3e292a5 (12.12)
