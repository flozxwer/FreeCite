# -*- coding: utf-8 -*-
# Script to evaluate the ability of a large language model to generate correct citations.
# This script currently only targets open-source models 
# and will use a function to evaluate whether the large language model generates correct citations.

from itertools import islice
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import requests
import re
import random
import time
import concurrent.futures
import threading
from tqdm import tqdm

# set random seed
random.seed(30)

print('check gpu： ',torch.cuda.is_available())

# Load reference data
fname='' # filename of data
with open(fname, 'r', encoding='utf-8') as file:
    data_citation_combo = json.load(file)

# handle prompt
def process_prompt(prompt):
    prompt_here=prompt
    prompt_here=prompt_here.replace('[Cite-*]', '[XXXXXXXX]')
    prompt_here=prompt_here.replace('[Cite-99]', '[11kk254v]')
    prompt_here=prompt_here.replace('[Cite-100]', '[2hdj4OHk]')
    pre_string='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    if "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你是一个中文大语言模型。" in prompt_here:
        # Preliminary instructions for question and answer data
        pre_instruction='我将提供给你一个问题，回答此问题可能需要的参考资料以及某AI助手对该问题的前半部分回答。请根据这些内容判断AI助手生成这前半部分回答的最后一句时使用了参考资料中的哪一条。\n'
        # Post instruction
        post_instruction='\n\nAI助手对此问题的前半部分回答：'
        
        # Question answering instruction removal
        chunk_ref=prompt_here.partition("\n\n参考资料：\n")[-1].partition("相关问答：")[0].partition("提示思路：")[0].partition("\n\n\n结构化模版：\n")[0]
        lis_here=prompt_here.partition(chunk_ref)
        pre_ref=lis_here[0]
    
        start_index = pre_ref.find("问题: ") + len("问题: ")
        end_index = pre_ref.find("\n\n补充信息：")
        question=pre_ref[start_index:end_index]
        pre_ref_new=pre_string+pre_instruction+"问题: "+question+"\n\n参考资料：\n\n"+chunk_ref+post_instruction
    else:
        # News data precommand
        pre_instruction='我将提供给你一个综述题目，创作此综述可能需要的参考资料以及某AI助手根据此题目创作的前半部分综述。请根据这些内容判断AI助手生成前半部分综述的最后一句时使用了参考资料中的哪一条。\n'
        # Post instruction
        post_instruction='\n\nAI助手根据此题目创作的前半部分综述：\n'
        # News order removal
        chunk_ref=prompt_here.partition("\n\n参考资料：\n")[-1].partition("注意遵守以下事项：\n1. 你需要在回答结果中插入引用证据的来源编号，格式为[编号]")[0]
        lis_here=prompt_here.partition(chunk_ref)
        pre_ref=lis_here[0]
                    
        start_index = pre_ref.find("\n你需要撰写的章节的分标题为：") + len("\n你需要撰写的章节的分标题为：")
        end_index = pre_ref.find("\n\n我将给你一些参考资料")
        question=pre_ref[start_index:end_index]
        pre_ref_new=pre_string+pre_instruction+"综述题目:"+question+"\n\n参考资料：\n\n"+chunk_ref+post_instruction
    return pre_ref_new


# post_mounted form prompt
def generate_post_mounted_prompt(pro_prompt):
    div_string='<|im_end|>\n<|im_start|>assistant\n'
    answer=pro_prompt.partition(div_string)[-1]
    processed_prompt = process_prompt(pro_prompt)
    post_mounted_prompt= processed_prompt+ "'"+ answer[:-1] + "'" + "\n\n你的选择应当以python8位字符串的形式输出，如'abcd1234', 'efgh5678'\n\nAI助手输出这句话所参考的资料编码可能是：\n" + div_string
    return post_mounted_prompt

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
            # Qwen2.5_7B，Qwen2.5_14B，Qwen2.5_32B，Qwen2.5_72B，Qwen2_7B, Qwen1.5_7B
            payload = json.dumps({
                'text': prompt,
                "model_name":"Qwen2.5_72B",
                "temperature":0,
                "max_tokens":50
                })
            # Send a request and get the response.
            response = requests.request("POST", url, headers=headers, data=payload,timeout=60)
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
        

def item_processing(dic:dict):
    prompt = dic['prompt']
    category = dic['category']
    output = dic['output']
    #print(single_sentences)
    post_mounted_prompt = generate_post_mounted_prompt(pro_prompt=prompt)
    try:
        ans_raw = citation_generation(post_mounted_prompt)['response']
    except (TypeError, KeyError) as e:
        return {
            "category": category,
            "output": output,
            "prompt": post_mounted_prompt,
            "response": "{}: {}".format(type(e).__name__, e)
        }
    dic_new={
        'category': category,
        'output': output,
        'prompt': post_mounted_prompt,
        'response': ans_raw
    }
    return dic_new

#print(citation_generation(prompt='请进行简单的自我介绍。'))
lock = threading.Lock()

f_res=""
# Initializes an empty list for storing dictionaries
qwen25_res = []
# Open the jsonl file and read it line by line
with open(f_res, 'r', encoding='utf-8') as file:
    for line in file:
        # Parse each line into a dictionary and add it to the list
        dict_data = json.loads(line)
        qwen25_res.append(dict_data)

print('qwen post-mounted format result data length',len(qwen25_res))

output_file = "" # final output
print(qwen25_res[0])
def process_list_and_write_to_file(data_list: list, output_file: str):
    # Create a new list to store the final results
    num = 0 
    handling_error_num=0

    for item in tqdm(data_list, desc="Processing items"):
        if item.get("response") != "RunTimeError Message\n\nFailed to get a response from the server.":
            # If the response is not an error message, add it directly to the result list
            num+=1
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False)+"\n")
        else:
             # If it is an error message, call the item_processing function until the response is no longer an error message
            attempt_count = 0  # the number of processing attempts for the current element
            last_result = item  # Stores the result of the last call
            while attempt_count < 3:
                last_result = item_processing(last_result)  # call item_processing function
                attempt_count += 1
                if last_result.get("response") != "RunTimeError Message\n\nFailed to get a response from the server.":
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(last_result, ensure_ascii=False)+"\n")
                    num += 1
                    handling_error_num += 1
                    break
            else:
                # If after 3 attempts, the result of the last call is added to the list of results
                with open(output_file, 'a', encoding='utf-8') as f:
                    json.dump(last_result, f, indent=4, ensure_ascii=False)
                    f.write(',')
                num += 1
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(']')

    print(f"Processing complete, successfully processed {num} elements.")
    print(f"Processing complete, successfully handled {handling_error_num} RunTimeErrors.")

process_list_and_write_to_file(qwen25_res, output_file)
