# coding:utf-8
import json
import requests
from tqdm import tqdm
import re
import concurrent.futures
import threading

key = ""
model = "gpt-4o"

key_dict={
    "gpt-4o": "",
    "baichuan":"",
    "moonshot": "",
    "doubao": "",
    "deepseek_v3":"",
    "glm":""
}
url_dict={
    "gpt-4o": "http://*/v1/chat/completions",
    "baichuan":"https://api.baichuan-ai.com/v1/chat/completions",
    "moonshot": "https://api.moonshot.cn/v1/chat/completions",
    "doubao": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    "deepseek_v3":"https://api.deepseek.com/chat/completions",
    "glm":"https://open.bigmodel.cn/api/paas/v4/chat/completions"
}
model_name_dict={
    "moonshot":"moonshot-v1-32k",
    "doubao":"ep-20250123113406-7vzr5",
    "deepseek_v3":"deepseek-chat",
    "glm":"glm-4-plus",
    "baichuan":"baichuan4-turbo"
}

fname='' # filename of data
with open(fname, 'r', encoding='utf-8') as file:
    prompt_for_artificial_data=json.load(file)

def chat_with_api(query: str,
                  key: str ,
                  url: str ,
                  model: str ,
                  system_message: str = None,
                  temperature: float = 0,
                  retry_time: int = 5,
                  json_mode: bool = False
                  ):
    if system_message:
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
    else:
        message = [
            {
                "role": "user",
                "content": query
            }
        ]
    payload = {
        "model": model,
        "messages": message,
        "temperature": temperature
    }
    if json_mode:
        payload.update(response_format={"type": "json_object"})
    payload = json.dumps(payload)
    headers = {
        'Authorization': 'Bearer {}'.format(key),
        'Content-Type': 'application/json',
    }
    count = 0
    while True:
        try:
            response = requests.request("POST", url, headers=headers, data=payload, timeout=300)
            print('response text:\n',response.text)
            result = json.loads(response.text)["choices"][0]["message"]["content"]
            break
        except Exception as e:
            count = count + 1
            print(e)
            if count > retry_time:
                return "RunTimeError Message"
    return result

# Process each raw citation data, where each raw citation data is a dictionary with three fields: category, label_prompt, and output.
# Return each raw citation data in a dictionary, including the original category, prompt, output, along with the model's response.
def item_processing(dic:dict):
    prompt = dic['label_prompt']
    
    # Delete the template of the Qianwen model.
    new_prompt = prompt.partition("<|im_end|>\n<|im_start|>user\n")[-1].partition("<|im_end|>\n<|im_start|>assistant\n")[0]

    category = dic['category']
    output= dic['output']
    try:
        ans_raw = chat_with_api(query=new_prompt,
                        key=key_dict['baichuan'],
                        url=url_dict['baichuan'],
                        model=model_name_dict['baichuan'])
    except (TypeError, KeyError) as e:
        return {
            "category": category,
            "output": output,
            "new prompt": new_prompt,
            "response": "{}: {}".format(type(e).__name__, e)
        }
    dic_new={
        'category': category,
        'new prompt': new_prompt,
        'response': ans_raw
    }
    return dic_new

lock = threading.Lock()
def parallel_processing(items):
    # Write square brackets when creating the file.
    filename_='' # filename of response
    with open(filename_, "a", encoding="utf-8") as f:
        f.write("[")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for result in tqdm(executor.map(item_processing, items), total=len(items)):
            with lock:
                #print(result)
                with open(filename_, 'a', encoding='utf-8') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
                    f.write(',')

    with open(filename_, 'a', encoding='utf-8') as f:
        f.write(']')
        
if __name__ == "__main__":
    print(chat_with_api(query='请进行简单自我介绍.',
                        key=key_dict['baichuan'],
                        url=url_dict['baichuan'],
                        model=model_name_dict['baichuan']))
    
    print('name of the model running:',model_name_dict['baichuan'])
    parallel_processing(prompt_for_artificial_data)
