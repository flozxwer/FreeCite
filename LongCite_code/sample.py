import requests
from utils.retrieve import  text_split_by_punctuation
import traceback
import json
import random
from transformers import AutoTokenizer

#测试prompt长度
def process_prompt(js):
    with open('one_shot_prompt.txt', "r", encoding="utf-8") as fp:
        prompt_format = fp.read()
    try:
        context, query = js['context'], js['query']
        sents = text_split_by_punctuation(context, return_dict=True)

        passage = ""
        for i, c in enumerate(sents):
            st, ed = c['start_idx'], c['end_idx']
            assert c['content'] == context[st:ed], c
            ed = sents[i+1]['start_idx'] if i < len(sents)-1 else len(context)
            passage += f"<C{i}>"+context[st:ed]
        
        prompt = prompt_format.replace('<<context>>', passage).replace('<<question>>', query)
        model_name = 'Qwen2_72B'
        # 准备发送到API的数据
        data = {
            'text': prompt,
            'model_name': model_name
        }
        url = ""
        response = requests.post(url, json=data)

        response_further = json.loads(response.text)

        prompt_len = response_further['prompt_len']
        #print("prompt_len",prompt_len)
        return prompt_len
    except:
        print("prompt construction error")
        traceback.print_exc()
        return None

#测试返回结果长度   
def remote_tokenize(text, model_name="Qwen2_72B"):
    # 远程服务的 API 地址
    url = ""
    
    # 构造请求数据
    payload = {
        "text": text,
        "model_name": model_name
    }
    
    # 发送 POST 请求
    response = requests.post(url, json=payload)
    response_further = json.loads(response.text)

    prompt_len = response_further['prompt_len']
    print("output_len",prompt_len)
#选取测试数据100条
def select_and_sample_data():
    # 读取数据
    with open("/path/to/LongBench-Cite.json", "r", encoding="utf-8") as file:
        ipts = json.load(file)
    
    # 筛选 prompt 长度在 31k 以下的数据
    filtered_data = []
    for js in ipts:
        prompt_len = process_prompt(js)
        if prompt_len is not None and prompt_len <= 31000:  # 假设单位是字符或 tokens
            print(prompt_len)
            filtered_data.append(js)
    
    # 随机选择 100 条数据
    if len(filtered_data) >= 100:
        sampled_data = random.sample(filtered_data, 100)
    else:
        sampled_data = filtered_data  # 如果不足 100 条，返回所有符合条件的数据
        print(f"警告：符合条件的数据只有 {len(filtered_data)} 条，不足 100 条")
    
    # 保存到 json 文件
    with open("100sample.json", "w", encoding="utf-8") as file:
        json.dump(sampled_data, file, ensure_ascii=False, indent=4)
    
    return sampled_data
    
if __name__ == '__main__':
    select_and_sample_data()
    with open("/path/to/100sample.json", "r", encoding="utf-8") as file:
        ipts = json.load(file)
# 初始化一个列表，用于存储提示符长度小于30k的数据的下标
    valid_indices = []
    sum=0
    sum_token=0
    for index, js in enumerate(ipts):
        try:
        # 调用process_prompt函数计算当前数据项的提示符长度
            prompt_len = process_prompt(js)
            sum_token+=prompt_len
            print(js["idx"])
            print(prompt_len)
        # 判断提示符长度是否小于30k
            if prompt_len is not None and prompt_len < 31000:
                valid_indices.append(index)
                sum+=1
        except:
            print(f"Error processing index {index}")
            traceback.print_exc()

    # 打印出所有符合条件的下标
    print("Indices of data with prompt length less than 30k:", sum)
    print("sum_token",sum_token)







