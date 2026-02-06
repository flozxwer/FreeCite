# -*- coding: utf-8 -*-
# Script to evaluate the ability of a large language model to generate correct citations.
# This script currently only targets open-source models 
# and will use a function to evaluate whether the large language model generates correct citations.

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import requests
import random
import time
import concurrent.futures
import threading
from tqdm import tqdm
# set random seed
random.seed(30)

print('check gpu: ',torch.cuda.is_available())

# Load reference data
fname='' # filename of data
with open(fname, 'r', encoding='utf-8') as file:
    data_citation_combo = json.load(file)


prompts=[item['prompt'] for item in data_citation_combo]
print('prompts count:',len(prompts))
print('unique prompts count:',len(set(prompts)))

max_retries = 3

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
            # Llama3.3_70B，glm_4_9B_chat，deepseek
            payload = json.dumps({
                'text': prompt,
                "model_name":"Qwen2_7B",
                "temperature":0,
                "max_tokens":20
                })
            # Send a request and get the response.
            response = requests.request("POST", url, headers=headers, data=payload,timeout=5)
            # Check the response status.
            response.raise_for_status()  # If the response is incorrect, throw an exception.
        
            if response.status_code == 200:
                # print('response text:\n',response.text)
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
                print('try number:',count)
                print("response.status_code:{}, response.text:{}".format(response.status_code, response.text))
                if count>= max_retries:
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
            

print("\nModel call started\n")
cnt_irr=0

# Process each raw citation data, where each raw citation data is a dictionary with three fields: category, label_prompt, and output.
# Return each raw citation data in a dictionary, including the original category, prompt, output, along with the model's response.
def item_processing(dic:dict):
    prompt = dic['prompt']
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
    end_index = ans_raw.find(']')
    ans_final = ans_raw[0:end_index]
    dic_new={
        'category': category,
        'output': output,
        'prompt': prompt,
        'response': ans_final
    }
    return dic_new
    

lock = threading.Lock()

def parallel_processing(items):
    # Write square brackets when creating the file.
    filename_='' # filename of response
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

# For debugging
# def parallel_processing_test(items):
#     with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
#         for result in tqdm(executor.map(item_processing, items), total=len(items)):
#             with lock:
#                 print(result)

parallel_processing(data_citation_combo)
#parallel_processing_test(data_citation_combo[1921:1922])
