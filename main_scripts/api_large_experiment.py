# Script used to call the closed-source model through its api
# coding:utf-8
import json
import sys
import requests
from tqdm import tqdm
import concurrent.futures
import threading
import random
import time


key_dict={
    "gpt": "",
    "baichuan":"",
    "moonshot": "",
    "doubao": "",
    "deepseek_v3":"",
    "glm":""
}

url_dict={
    "gpt": "https://apigateway.offline.xinyunews.cn/llm/v1/chat/completions",
    "baichuan":"https://api.baichuan-ai.com/v1/chat/completions",
    "moonshot": "https://api.moonshot.cn/v1/chat/completions",
    "doubao": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    "deepseek_v3":"https://api.deepseek.com/chat/completions",
    "glm":"https://open.bigmodel.cn/api/paas/v4/chat/completions"
}

# gpt-4-turbo    gpt-4o    gpt-4o-mini 
model_name_dict={
    "gpt":"gpt-4o",
    "moonshot":"moonshot-v1-32k",
    "doubao":"ep-20250123113406-7vzr5",
    "deepseek_v3":"deepseek-chat",
    "glm":"glm-4-plus",
    "baichuan":"baichuan4-turbo"

}
# the name of the model used in this experiment
model_name_here='gpt'
fname='' # filename of data

with open(fname, 'r', encoding='utf-8') as file:
    data_citation_combo = json.load(file)

prompts=[item['prompt'] for item in data_citation_combo]
print('prompts count:',len(prompts))
print('unique prompts count:',len(set(prompts)))

def chat_with_api(user_msg: str,
                  assistant_msg: str,
                  key: str ,
                  url: str ,
                  model: str ,
                  system_message: str = None,
                  temperature: float = 0,
                  retry_time: int = 6,
                  json_mode: bool = False
                  ):
    if system_message:
        query = "<im_user>{}<user_end><im_assistant>{}".format(user_msg, assistant_msg)
        message = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": query
            }
        ]
    else:
        system_message = "我将给你提供某AI助手与其用户的一段对话，其中AI助手的发言被截断了一部分，请根据上下文语境进行补充。注意：用户的发言以'<im_user>'开始，以'<user_end>'结束，AI助手的发言以'<im_assistant>'开始。"
        query = "<im_user>{}<user_end><im_assistant>{}".format(user_msg, assistant_msg)
        message = [   
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": query
            },
        ]
    payload = {
        "model": model,
        "messages": message,
        "temperature": temperature,
        "max_tokens":20
    }
    if json_mode:
        payload.update(response_format={"type": "json_object"})
    payload = json.dumps(payload)
    headers = {
        'Authorization': 'Bearer {}'.format(key),
        'Content-Type': 'application/json',
    }
    count = 0
    response = None  # init response
    while True:
        try:
            response = requests.request("POST", url, headers=headers, data=payload, timeout=300)
            if response.status_code == 200:
                # print('response text:\n',response.text)
                result = json.loads(response.text)["choices"][0]["message"]["content"]
                res = {
                    'prompt': query,
                    'response': result
                }
                break
            else:
                count=count+1
                print('try number:',count)
                print("response.status_code:{}, response.text:{}".format(response.status_code, response.text))
                if count>= retry_time:
                    res = {
                        'prompt': query,
                        'response':  "RunTimeError Message\n\n" + response.text,
                    }
                    return res
                time.sleep(1)
        except Exception as e:
            count = count + 1
            print('try number:',count)
            print("response.status_code:{}, response.text:{}".format(response.status_code, response.text))
            print("Full error is {}, full response is {}".format(e, response))
            if count >= retry_time:
                if response:
                    result = "RunTimeError Message\n\n" + response.text
                    res = {
                        'prompt':query,
                        'response':result
                    }
                else:
                    result = "RunTimeError Message\n\nFailed to get a response from the server."
                    res = {
                        'prompt':query,
                        'response':result,
                    }
                return res
    return res

# Process each raw citation data, where each raw citation data is a dictionary with three fields: category, label_prompt, and output.
# Return each raw citation data in a dictionary, including the original category, prompt, output, along with the model's response.
def item_processing(dic:dict):
    # Add a random sleep between 0 and 2 seconds
    time.sleep(random.uniform(0, 2))
    prompt = dic['prompt']
    # Question-answer data and news data with some instructions removed.
    str_to_replace_QA1='\n结构化模版：\n为了使答案更清晰和有组织，以下是几种常见的结构化方式，你可以选用其中一种或多种方式组织答案：\n-简介-主体-总结：引入主题，详细讨论，以要点总结。\n-针对每个子问题的段落：适合复杂问题，每一子问题一个段落回答。\n-因果关系：说明事件的原因和结果。\n-比较对比：描述并对比两个以上的概念或事物。\n-时间顺序：按事件发生的顺序描述过程或步骤。\n-问题解决：介绍问题，阐述解决方案和策略。\n-优缺点：列举决策或选择的正反两面。\n-定义和例子：给出定义并通过例子解释。\n-逻辑推理：基于假设或前提，逻辑推导结论。\n-列表结构：列出事实或特点，方便扫描。\n-分类结构：介绍概念，按标准分组并详细说明。\n-主题和变化：探讨核心主题及其变体。\n-案例研究：通过具体案例解释理论或概念。\n-层次结构：信息按重要性或顺序排列。\n-议题和反议：展示议题的支持和反对观点。'
    str_to_div_QA2='\n\n另外遵循以下要求：'
    str_to_replace_news1='，你的输出应当以\"[综述]\"作为前缀，即。\n#############\n[综述]: XXXXXXX\n############# '
    str_to_replace_QA3='\n\n在结构化答案时，'
    str_new_QA3='\n\n在输出答案时，'

    # Split required user_message&assistant message
    user_message_here = prompt.partition("<|im_end|>\n<|im_start|>user\n")[-1].partition("<|im_end|>\n<|im_start|>assistant\n")[0] 
    assistant_message_here = prompt.partition("<|im_end|>\n<|im_start|>assistant\n")[-1]
    
    category = dic['category']
    output= dic['output']
    try:
        if "你是一个中文大语言模型。你在做一个百科问答任务，" in prompt:
            user_message_here=user_message_here.replace(str_to_replace_QA1,'').partition(str_to_div_QA2)[0].replace(str_to_replace_QA3,str_new_QA3)
            ans_raw = chat_with_api(user_msg=user_message_here,
                              assistant_msg=assistant_message_here,
                              key=key_dict[model_name_here],
                                url=url_dict[model_name_here],
                                model=model_name_dict[model_name_here],
                                json_mode=False)
        else:
            user_message_here=user_message_here.replace(str_to_replace_news1,'')
            assistant_message_here=assistant_message_here[5:]
            ans_raw = chat_with_api(user_msg=user_message_here,
                              assistant_msg=assistant_message_here,
                              key=key_dict[model_name_here],
                                url=url_dict[model_name_here],
                                model=model_name_dict[model_name_here],
                                json_mode=False)
    except (TypeError, KeyError) as e:
        return {
            "category": category,
            "output": output,
            "prompt": "<im_user>{}<user_end><im_assistant>{}".format(user_message_here, assistant_message_here),
            "response": "{}: {}".format(type(e).__name__, e)
        }
    dic_new={
        'category': category,
        "output": output,
        "prompt": ans_raw['prompt'],
        "response": ans_raw['response']
    }
    return dic_new

lock = threading.Lock()
def parallel_processing(items):
    # Write square brackets when creating the file.
    filename_='' # filename of response
    with open(filename_, "a", encoding="utf-8") as f:
        f.write("[")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for result in tqdm(executor.map(item_processing, items), total=len(items)):
            with lock:
                # print(result)
                with open(filename_, 'a', encoding='utf-8') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
                    f.write(',')

    with open(filename_, 'a', encoding='utf-8') as f:
        f.write(']')


# For debugging
# def parallel_processing_test(items):
#     with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#         for result in tqdm(executor.map(item_processing, items), total=len(items)):
#             with lock:
#                 print(result)

print('name of the model running',model_name_dict[model_name_here])  
# random_combo=random.sample(data_citation_combo,100)      
parallel_processing(data_citation_combo)
# parallel_processing_test(data_citation_combo[:1])
