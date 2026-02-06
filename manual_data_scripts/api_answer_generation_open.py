import json
import re
from tqdm import tqdm
import requests
import time
import random
import concurrent.futures
import threading
from tqdm import tqdm


fname='' # filename of data
with open(fname, 'r', encoding='utf-8') as file:
    prompt_for_artificial_data=json.load(file)

def citation_generation(prompt):
    url = ""

    headers = {
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Connection': 'keep-alive'
        }
    max_retries=3
    count = 0
    response = None  # init response
    while True:
        try:
            # model_name can be one of the following options.
            # Qwen2.5_7B，Qwen2.5_14B，Qwen2.5_32B，Qwen2.5_72B，Qwen2_7B，Qwen2_57B，Qwen2_72B
            # Llama3.3_70B
            payload = json.dumps({
                'text': prompt,
                "model_name":"Llama3.3-70B",
                "temperature":0,
                "max_tokens":2048
                })
            # Send a request and get the response.
            response = requests.request("POST", url, headers=headers, data=payload,timeout=300)
            #print(response.text)
            # Check the response status.
            response.raise_for_status()  # If the response is incorrect, throw an exception.
            if response.status_code == 200:
                #print('response text:\n',response.text)
                response_json = json.loads(response.text)
                if "text" in response_json:
                    result = response_json["text"][0].partition(prompt)[-1]
                res = {
                    'prompt': prompt,
                    'response': result
                }
                break
            else:
                count=count+1
                print("response.status_code:{}, response.text:{}".format(response.status_code, response.text))
                if count>=max_retries:
                    res = {
                        'prompt': prompt,
                        'response':  "RunTimeError Message\n\n" + response.text,
                    }
                    return res
                time.sleep(60)
        except Exception as e:
            count = count + 1
            print(f"Request failed: {e}, retrying... ({count}/{max_retries})")
            if count >= max_retries:
                if response:
                    result = "RunTimeError Message\n\n" + response.text
                    res = {
                        'prompt':prompt,
                        'response':result
                    }
                else:
                    result = "RunTimeError Message\n\nFailed to get a response from the server."
                    res = {
                        'prompt':prompt,
                        'response':result
                    }
                return res
    return res

# Process each raw citation data, where each raw citation data is a dictionary with three fields: category, label_prompt, and output.
# Return each raw citation data in a dictionary, including the original category, prompt, output, along with the model's response.
def item_processing(dic:dict):
    prompt = dic['label_prompt']
    #Llama3.3-70B
    # new_before_string='<s><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    # old_before_string='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    # prompt=prompt.replace(old_before_string,new_before_string)
    # new_div_string='<|im_end|>\n'
    # old_div_string='<|im_end|>\n<|im_start|>assistant\n'
    # prompt=prompt.replace(old_div_string,new_div_string)

    category = dic['category']
    output= dic['output']
    try:
        ans_raw = citation_generation(prompt)['response']
    except (TypeError, KeyError) as e:
        return {
            "category": category,
            "output": output,
            "prompt": prompt,
            "response": "{}: {}".format(type(e).__name__, e)
        }
    dic_new={
        'category': category,
        'prompt': prompt,
        'response': ans_raw
    }
    return dic_new

lock = threading.Lock()
def parallel_processing(items):
    # Write square brackets when creating the file.
    filename_='' # filename of reaponse
    with open(filename_, "a", encoding="utf-8") as f:
        f.write("[")
        f.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for result in tqdm(executor.map(item_processing, items), total=len(items)):
            with lock:
                with open(filename_, 'a', encoding='utf-8') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
                    f.write(',')

    with open(filename_, 'a', encoding='utf-8') as f:
        f.write(']')

parallel_processing(prompt_for_artificial_data[3:4])
