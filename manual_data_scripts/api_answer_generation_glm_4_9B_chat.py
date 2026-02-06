import json
import re
from tqdm import tqdm
import requests
import time
import random
import concurrent.futures
import threading
from tqdm import tqdm

# Load reference data
fname='' # filename of data
with open(fname, 'r', encoding='utf-8') as file:
    data_citation_combo = json.load(file)
prompts=[item['prompt'] for item in data_citation_combo]
print('prompts count:',len(prompts))
print('unique prompts count:',len(set(prompts)))
print('/....../....../....../....../....../')

# Input the concatenated version of the prompt, output the prompt for manual annotation, and modify the template
def generate_label_prompt(prompt_ori):
    # Remove the prompt after 'assistant' from the prompt.
    new_before_string='[gMASK]<sop><|system|>\nYou are a helpful assistant<|user|>\n'
    old_before_string='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    prompt=prompt_ori.replace(old_before_string,new_before_string)
    new_div_string='<|assistant|>\n'
    old_div_string='<|im_end|>\n<|im_start|>assistant\n'
    prompt=prompt.replace(old_div_string,new_div_string)
    div_string='<|assistant|>\n'
    
    label_prompt = prompt.partition(div_string)[0] + div_string

    return label_prompt


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
            # glm_4_9B_chat
            payload = json.dumps({
                'text': prompt,
                "model_name":"glm_4_9B_chat",
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

# Check if there are citations in the response.
def check_citation(response):
    """
    Check if the input string contains citations in the format of [12345678].
    para：
    - response: input String
    return：
    - If citations in this format exist, return True; otherwise, return False.
    """
    # Define the regular expression pattern.
    # Match the pattern like [12345678], where the numeric part can be of any length.
    pattern = r'\[([a-zA-Z0-9]{8})\]'
    # Use re.search to check if a match exists.
    if re.search(pattern, response):
        return True
    else:
        return False
    
# Delete the content in the string after the second occurrence of the substring.
def remove_from_second_occurrence(text, substring):
    # Find the position of the first occurrence.
    first_occurrence = text.find(substring)
    if first_occurrence == -1:
        return text  # If not found, return the original string directly
    
    # Continue searching for the second occurrence after the position of the first occurrence.
    second_occurrence = text.find(substring, first_occurrence + len(substring))
    if second_occurrence == -1:
        return text  # If the second occurrence is not found, return the original string directly.
    
    # Return the content from the beginning up to just before the second occurrence.
    return text[:second_occurrence]

# Process each raw citation data, where each raw citation data is a dictionary with three fields: category, label_prompt, and output.
# Return each raw citation data in a dictionary, including the original category, prompt, output, along with the model's response.
def item_processing(dic:dict):
    prompt = dic['label_prompt']
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
    if check_citation(ans_raw):
        if ans_raw.find("[回答]:")!=-1:
            ans_raw=remove_from_second_occurrence(ans_raw, "[回答]:")
        elif ans_raw.find("[综述]:")!=-1:
            ans_raw=remove_from_second_occurrence(ans_raw, "[综述]:")
        else:
            ans_raw="No citation"
        dic_new={
            'category': category,
            'prompt': prompt,
            'response': ans_raw
        }
    else:
        dic_new={
            'category': category,
            'prompt': prompt,
            'response':"No citation"
        }
    return dic_new

filename_=''  # filename of response
with open(filename_, "a", encoding="utf-8") as f:
    f.write("[")
    f.close()

# Remove duplicates and create a mapping.
dic_mapping = {}
for item in data_citation_combo:
    output = item['output']
    if output not in dic_mapping:
        dic_mapping[output] = [item]
    else:
        dic_mapping[output].append(item)

print('Number of chapters after processing:', len(dic_mapping.keys()))

# Convert all outputs into a list
outputs = list(dic_mapping.keys())

# Initialize the counter and result list
non_no_citation_count = 0
selected_outputs = []  # Record the selected chapters

pbar = tqdm(total=100, desc="Processing citations", unit="citation")
# Process until 100 responses that are not 'No citation' are obtained
while non_no_citation_count < 100:
    # Randomly select an output from the outputs, excluding those already marked as 'No citation' and the ones that have been selected.
    available_outputs = [o for o in outputs if o not in selected_outputs]
    if not available_outputs:
        break  # If there are no available outputs, exit the loop.
    
    output_key = random.choice(available_outputs)
    selected_outputs.append(output_key)  # Mark as selected.
    ite = dic_mapping[output_key]
    prompt = ite[0]['prompt']
    label_prompt = generate_label_prompt(prompt)
    category = ite[0]['category']
    dic = {
        "category": category,
        "output": output_key,
        "label_prompt": label_prompt
    }
    
    # call item_processing function
    result = item_processing(dic)
    
    # Check if the returned 'response' is 'No citation'.
    if result.get('response') != "No citation":
        with open(filename_, 'a', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
            f.write(',')
        non_no_citation_count += 1
        pbar.update(1)  # Update the progress bar

pbar.close()   
with open(filename_, 'a', encoding='utf-8') as f:
    f.write(']')
print("end*******************")

#parallel_processing(prompt_for_artificial_data[:10])
