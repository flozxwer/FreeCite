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

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Evaluate the ability of a large language model to generate correct citations.")

parser.add_argument('--model_name', type=str, default="Qwen2.5_7B", help='Model name (default: Qwen2.5_7B)')
args = parser.parse_args()

key_dict={
    "gpt-4o-2024-08-06": "",
    "gpt-4-turbo": "",
    "gpt-4o-mini": "",
    "deepseek": "",
    "moonshot": "",
    "doubao": "",
    "deepseek_v3":"",
    "glm":"",
    "baichuan":"",
    "qwen":"",
    "claude":"",
    "gemini":""
}

url_dict={
    "gpt-4o-2024-08-06": "",
    "gpt-4-turbo": "",
    "gpt-4o-mini": "",
    "deepseek": "",
    "baichuan":"https://api.baichuan-ai.com/v1/chat/completions",
    "moonshot": "https://api.moonshot.cn/v1/chat/completions",
    "doubao": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    "deepseek_v3":"https://api.deepseek.com/chat/completions",
    "glm":"https://open.bigmodel.cn/api/paas/v4/chat/completions",
    "qwen":"https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    "claude":"",
    "gemini":""
}

# gpt-4-turbo    gpt-4o    gpt-4o-mini 
model_name_dict={
    "gpt-4o-2024-08-06":"gpt-4o-2024-08-06",
    "gpt-4-turbo":"gpt-4-turbo",
    "gpt-4o-mini":"gpt-4o-mini",
    "deepseek":"deepseek-v3",
    "moonshot":"moonshot-v1-32k",
    "doubao":"ep-20250123113406-7vzr5",
    "deepseek_v3":"deepseek-chat",
    "glm":"glm-4-plus",
    "baichuan":"baichuan4-turbo",
    "qwen":"qwen2.5-7b-instruct",
    "claude":"claude-3-5-haiku-20241022",
    "gemini":"gemini-2.0-flash"
}
# the name of the model used in this experiment
model_name_here=args.model_name

fname='/../translated_closed_prompt_text.json' # filename of data

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
        "max_tokens":4096
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
    category = dic['category']
    output= dic['output']
    user_message_here=dic.get('translated info').get('user message')
    assistant_message_here=dic.get('translated info').get('assistant message')
    try:
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
            "prompt": prompt,
            "eng prompt": "<im_user>{}<user_end><im_assistant>{}".format(user_message_here, assistant_message_here),
            "response": "{}: {}".format(type(e).__name__, e)
        }
    dic_new={
        'category': category,
        "output": output,
        "prompt": prompt,
        "eng prompt": ans_raw['prompt'],
        "response": ans_raw['response']
    }
    return dic_new

lock = threading.Lock()
def parallel_processing(items):
    # Write square brackets when creating the file.
    filename_=f'/../res_api_model_eng_0730_{model_name_dict[args.model_name]}_new1.json' # filename of response
    # filename_='gpt-4o-mini_0730_eng_try_01.json'
    with open(filename_, "a", encoding="utf-8") as f:
        f.write("[")

    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
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
