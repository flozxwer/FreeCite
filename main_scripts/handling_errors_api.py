# Closed-source model handling error.

import json
import requests
import random
import time
from tqdm import tqdm

# set random seed
random.seed(30)

# Load reference data
fname='' # filename of data
with open(fname, 'r', encoding='utf-8') as file:
    content=file.read()
if content.endswith(',]'):
    content=content[:-2]+']'
if content.endswith(','):
    print('end with comma')
    content=content[:-1]+']'
data_citation_combo= json.loads(content)

output_file='' # filename of data handled

prompts=[item['prompt'] for item in data_citation_combo]
print('prompts count:',len(prompts))
print('unique prompts count:',len(set(prompts)))

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

# the name of the model used in this experiment.
model_name_here='gpt'

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
                time.sleep(60)
        except Exception as e:
            count = count + 1
            print('try number:',count)
            print("Full error is: {}, full response is: {}".format(e, response))
            print("response.status_code:{}, response.text:{}".format(response.status_code, response.text))
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
    user_message_here = prompt.partition("<im_user>")[-1].partition("<user_end><im_assistant>")[0] 
    assistant_message_here = prompt.partition("<user_end><im_assistant>")[-1]
    
    category = dic['category']
    output= dic['output']
    try:
        if "你是一个中文大语言模型。你在做一个百科问答任务，" in prompt:
            ans_raw = chat_with_api(user_msg=user_message_here,
                              assistant_msg=assistant_message_here,
                              key=key_dict[model_name_here],
                                url=url_dict[model_name_here],
                                model=model_name_dict[model_name_here],
                                json_mode=False)
        else:
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

    
def process_list_and_write_to_file(data_list: list, output_file: str):
    # Create a new list to store the final results.
    num = 0 
    handling_error_num=0
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("[")
        f.close()
    for item in tqdm(data_list, desc="Processing items"):
        if "RunTimeError Message\n\n" not in item.get("response"):
        #if item.get("response") != "RunTimeError Message\n\nFailed to get a response from the server":
            # If the response is not an error message, add it directly to the result list.
            num+=1
            with open(output_file, 'a', encoding='utf-8') as f:
                json.dump(item, f, indent=4, ensure_ascii=False)
                f.write(',')
        else:
             # If the response is not an error message, add it directly to the result list.
            attempt_count = 0  # the number of attempts for processing the current element
            last_result = item  # Store the result of the last call.
            while attempt_count < 3:
                last_result = item_processing(last_result)  # call item_processing function
                attempt_count += 1
                if "RunTimeError Message\n\n" not in last_result.get("response"):
                #if last_result.get("response") != "RunTimeError Message\n\nFailed to get a response from the server":
                    with open(output_file, 'a', encoding='utf-8') as f:
                        json.dump(last_result, f, indent=4, ensure_ascii=False)
                        f.write(',')
                    num += 1
                    handling_error_num += 1
                    break
            else:
                # If the attempt fails after 3 tries, add the result of the last call to the result list.
                with open(output_file, 'a', encoding='utf-8') as f:
                    json.dump(last_result, f, indent=4, ensure_ascii=False)
                    f.write(',')
                num += 1
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(']')

    print(f"Processing complete, successfully processed {num} elements.")
    print(f"Processing complete, successfully handled {handling_error_num} RunTimeErrors.")

# For debugging
def process_list_and_write_to_file_test(data_list: list):
    num = 0 
    handling_error_num=0
    for item in tqdm(data_list, desc="Processing items"):
        print("current item：",item)
        #print("response of current item：",item.get("response"))
        if "RunTimeError Message\n\n" not in item.get("response"):
            num+=1
        else:
            print("start handling RunTimeError")
            attempt_count = 0
            last_result = item
            while attempt_count < 3:
                last_result = item_processing(last_result)
                attempt_count += 1
                if "RunTimeError Message\n\n" not in last_result.get("response"):
                    num += 1
                    handling_error_num += 1
                    print(last_result)
                    break
            else:
                num += 1
    print(f"Processing complete, successfully processed {num} elements.")
    print(f"Processing complete, successfully handled {handling_error_num} RunTimeErrors.")

process_list_and_write_to_file(data_citation_combo[7313:], output_file)
#process_list_and_write_to_file_test(data_citation_combo[7556:7557])
