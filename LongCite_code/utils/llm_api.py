import requests
import time
import os, json
from tqdm import tqdm
import traceback

API_KEYS = {
    # "openai": '',
    # "zhipu": '',
    # "anthropic": '',
    # 'vllm': 'token-abc123',
    "open": '',
    "gpt": "",
    "baichuan":"",
    "moonshot": "",
    "doubao": "",
    "deepseek_v3":"",
    "glm-4-plus":""
}

API_URLS = {
    # 'openai': 'https://api.openai.com/v1/chat/completions',
    # 'zhipu': 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
    # "anthropic": 'https://api.anthropic.com/v1/messages',
    # 'vllm': 'http://127.0.0.1:8000/v1/chat/completions',
    'open': '',
    "gpt": "",
    "baichuan":"https://api.baichuan-ai.com/v1/chat/completions",
    "moonshot": "https://api.moonshot.cn/v1/chat/completions",
    "doubao": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    "deepseek_v3":"",
    #"glm":"https://open.bigmodel.cn/api/paas/v4/chat/completions"
}

model_name_dict={
    "gpt":"gpt-4o", # gpt-4o, gpt-4-turbo, gpt-4o-mini
    "moonshot":"moonshot-v1-32k",
    "doubao":"ep-20250123113406-7vzr5",
    "deepseek_v3":"deepseek-v3",
    #"glm":"glm-4-plus",
    "baichuan":"baichuan4-turbo"
}

def query_llm(messages, model, temperature=1.0, max_new_tokens=1024, stop=None, return_usage=False):
    # if 'gpt' in model:
    #     api_key, api_url = API_KEYS['openai'], API_URLS['openai']
    # elif 'glm' in model:
    #     api_key, api_url = API_KEYS['zhipu'], API_URLS['zhipu']
    # elif 'claude' in model:
    #     api_key, api_url = API_KEYS['anthropic'], API_URLS['anthropic']
    # else:
    #     api_key, api_url = API_KEYS['vllm'], API_URLS['vllm']

    if 'gpt' in model:
        api_key, api_url = API_KEYS['gpt'], API_URLS['gpt']
    elif 'baichuan' in model:
        api_key, api_url = API_KEYS['baichuan'], API_URLS['baichuan']
    elif 'moonshot' in model:
        api_key, api_url = API_KEYS['moonshot'], API_URLS['moonshot']
    elif 'doubao' in model:
        api_key, api_url = API_KEYS['doubao'], API_URLS['doubao']
    elif 'deepseek_v3' in model:
        api_key, api_url = API_KEYS['deepseek_v3'], API_URLS['deepseek_v3']
    # elif 'glm' in model:
    #     api_key, api_url = API_KEYS['glm'], API_URLS['glm']
    else:
        api_key, api_url = API_KEYS['open'], API_URLS['open']

    if model in model_name_dict:
        model = model_name_dict[model]
    
    tries = 0
    while tries < 5:
        tries += 1
        try:
            if api_key == API_KEYS['open']:
                headers = {
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json',
                    'Accept': '*/*',
                    'Connection': 'keep-alive'
                }
                resp = requests.post(api_url, json = {
                    "text": messages,
                    "model_name": model,
                    "temperature": temperature,
                    "max_tokens": max_new_tokens,
                    "stop" if 'claude' not in model else 'stop_sequences': stop,
                }, headers=headers, timeout=600)
            else:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}" if api_key else None
                }
                resp = requests.post(api_url, json = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_new_tokens,
                    "stop" if 'claude' not in model else 'stop_sequences': stop,
                }, headers=headers, timeout=600)
            # if 'claude' not in model:
            #     headers = {
            #         'Authorization': "Bearer {}".format(api_key),
            #     }
            # else:
            #     headers = {
            #         'x-api-key': api_key,
            #         'anthropic-version': "2023-06-01",
            #     }   
            # resp = requests.post(api_url, json = {
            #     "model": model,
            #     "messages": messages,
            #     "temperature": temperature,
            #     "max_tokens": max_new_tokens,
            #     "stop" if 'claude' not in model else 'stop_sequences': stop,
            # }, headers=headers, timeout=600)
            # print(resp.text)
            # print(resp.status_code)
            if resp.status_code != 200:
                raise Exception(resp.text)
            resp = resp.json()
            break
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if "maximum context length" in str(e):
                raise e
            elif "triggering" in str(e):
                return 'Trigger OpenAI\'s content management policy.',resp.status_code if 'resp' in locals() else -1
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return None
    try:
        # 开源模型返回格式
        if "text" in resp:
            return resp["text"][0]
        if 'content' not in resp["choices"][0]["message"] and 'content_filter_results' in resp["choices"][0]:
            resp["choices"][0]["message"]["content"] = 'Trigger OpenAI\'s content management policy.'
        if return_usage:
            return resp["choices"][0]["message"]["content"], resp['usage']
        else:
            return resp["choices"][0]["message"]["content"]
    except: 
        return None

if __name__ == '__main__':
    model = 'Qwen2.5_72B'
    is_open = True # 调用开源模型则设为True
    prompt = '你是谁'
    if is_open :
        msg = prompt
    else:
        msg = [{'role': 'user', 'content': prompt}]
    output = query_llm(msg, model=model, temperature=1, max_new_tokens=10, stop=None, return_usage=True)
    print(output)
