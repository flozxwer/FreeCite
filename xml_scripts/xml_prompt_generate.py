import json
import re
from tqdm import tqdm
import requests
import time
import random

import concurrent.futures
import threading


# Load reference data
fname='../3000_sample.json'
with open(fname, 'r', encoding='utf-8') as file:
    data_citation_combo = json.load(file)

prompts=[item['prompt'] for item in data_citation_combo]
print('prompts count:',len(prompts))
print('unique prompts count:',len(set(prompts)))

# Process the answer from the beginning to the sentence with a matching citation, modified to xml format.
# It needs to be consistent with other formats.
def process_post_answer(answer_here):
    # Define a regular expression to match the citation ID
    pattern = r'(\[[A-Za-z0-9]{8}\])' # slice the answer and matches the citation ID obtained from the slice
    pattern_2 = r'\[([A-Za-z0-9]{8})\]' # replace the sliced citation ID in the new format
    # Use re.split to split the string, keeping the citation ID
    split_text = re.split(pattern, answer_here[:-1])

    # Filter out empty strings
    split_text = [part for part in split_text if part.strip()]

    if ''.join(split_text) != answer_here[:-1]:
        print('String alignment test error!')
        print(answer_here)

    # xml format modification
    # xml format changes the first sentence
    replaced_texts=[]

    txt=split_text[0]
    replaced_text_first = """
    <cited_answer>
        <answer>{}</answer>
        <citations>
""".format(txt)
    replaced_texts.append(replaced_text_first)

    cnt_ids=0
    for splited_str in split_text[1:]:
        # Replace the subsequent cut out citation id with xml format
        if re.search(pattern, splited_str):
            replaced_text = re.sub(pattern_2,r'            <source_id>\1</source_id>\n',splited_str)
            replaced_texts.append(replaced_text)
            cnt_ids=cnt_ids+1
        # Replace the subsequent segmented sentences with xml format
        else:
            replaced_text="        </citations>\n        <answer>"+splited_str+"</answer>\n        <citations>\n"
            replaced_texts.append(replaced_text)

    new_post_answer=''.join(replaced_texts)+"            <source_id>"
    ori_ids = len(re.findall(pattern,answer_here))
    if cnt_ids!=ori_ids:
        print('id number mismatch error!')
    return new_post_answer

# Input the spliced version prompt and output the xml format prompt:
def generate_xml_prompt(prompt_ori):
    # Extract the reference section, the citation id corresponds to the data one by one, and store it
    if "你是一个中文大语言模型。你在做一个百科问答任务，" in prompt_ori:
        raw_ref_chunk=prompt_ori.partition("\n\n参考资料：\n")[-1].partition("相关问答：")[0].partition("提示思路：")[0].partition("\n\n\n结构化模版：\n")[0]
    else:
        raw_ref_chunk = prompt_ori.partition("\n\n参考资料：\n")[-1].partition("注意遵守以下事项：\n1. 你需要在回答结果中插入引用证据的来源编号，格式为[编号]")[0]
    pattern_match=re.compile(r'\[([a-zA-Z0-9]{8})\]') # Finds all citation ids
    pattern_sep = re.compile(r'(\[([a-zA-Z0-9]{8})\])(.*?)(?=\[([a-zA-Z0-9]{8})\]|\Z)', re.DOTALL) # Finds all citation ids and corresponding citation data
    all_refs = pattern_sep.findall(raw_ref_chunk) # All citation ids
    all_marks=pattern_match.findall(raw_ref_chunk) # All citation ids and corresponding references
    if len(all_marks)!=len(all_refs):
        print('The number of references does not match correctly')
    # Convert the references to xml format
    new_ref_chunk='<references>\n'
    for ite in all_refs:
        ref_id=ite[1]
        ref_content=ite[2][1:].rstrip('\n')
        formated_single_cite=f"    <reference><source_id>{ref_id}</source_id><content>{ref_content}</content></reference>\n"
        new_ref_chunk=new_ref_chunk+formated_single_cite
    new_ref_chunk=new_ref_chunk+'</references>\n\n'
    # Extract the problem and generate a simplified version of the reference before insturction
    if "你是一个中文大语言模型。你在做一个百科问答任务，" in prompt_ori:
        pre_ref=prompt_ori.partition(raw_ref_chunk)[0]
        start_index = pre_ref.find("问题: ") + len("问题: ")
        end_index = pre_ref.find("\n\n补充信息：")
        question=pre_ref[start_index:end_index]
        pre_ref_new='你在做一个百科问答任务，请基于参考资料来回答问题。\n\n'+"问题: "+question+"\n\n参考资料：\n\n"
    else:
        pre_ref=prompt_ori.partition(raw_ref_chunk)[0]
        start_index = pre_ref.find("\n你需要撰写的章节的分标题为：") + len("\n你需要撰写的章节的分标题为：")
        end_index = pre_ref.find("\n\n我将给你一些参考资料")
        question=pre_ref[start_index:end_index]
        pre_ref_new='你在创作一篇综述。请基于参考资料和给定的标题来创作这篇综述。\n\n'+"标题: "+question+"\n\n参考资料：\n\n"
    # The instruction after the reference is generated
    div_string='<|im_end|>\n<|im_start|>assistant\n'
    # Matches include answers or summaries with cited sentences. 
    # From the beginning of the answer or summary to the sentence to be matched
    matched_answer=prompt_ori.partition(div_string)[-1] # not removing the left parenthesis after matching
    # Generate answer in xml format
    xml_answer=process_post_answer(answer_here=matched_answer)

    post_ref="""请按照如下格式输出:
    <cited_answer>
        <answer></answer>
        <citations>
            <source_id></source_id>
            <source_id></source_id>
            ...
        </citations>
    </cited_answer>\n\n"""+"""回答示例:
    <cited_answer>
        <answer>习近平总书记在2023年全国两会期间的行程安排紧凑，涉及多个重要活动和会议</answer>
        <citations>
            <source_id>11kk254v</source_id>
            <source_id>2hdj4OHk</source_id>
        </citations>
    </cited_answer>\n\n"""+div_string+xml_answer
    # Concatenate the final xml format prompt
    xml_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"+pre_ref_new+new_ref_chunk+post_ref
    return xml_prompt
print("---------","\n\n\n\n\n\n")

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
            

print("\nModel call start\n")
cnt_irr=0

# Process each raw citation data, where each raw citation data is a dictionary with three fields: category, label_prompt, and output.
# Return each raw citation data in a dictionary, including the original category, prompt, output, along with the model's response.
def item_processing(dic:dict):
    prompt = dic['prompt']
    category = dic['category']
    output= dic['output']
    xml_prompt = generate_xml_prompt(prompt)
    try:
        ans_raw = citation_generation(xml_prompt)['response']
    except (TypeError, KeyError) as e:
        return {
            "category": category,
            "output": output,
            "prompt": xml_prompt,
            "response": "{}: {}".format(type(e).__name__, e)
        }
    #end_index = ans_raw.find(']')
    end_index = ans_raw.find('</source_id>')
    ans_final = ans_raw[:end_index]
    dic_new={
        'category': category,
        'output': output,
        'prompt': xml_prompt,
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

parallel_processing(data_citation_combo)
