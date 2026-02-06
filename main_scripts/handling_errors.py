# Open-source model handling error.

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
            # The model_name can be one of the following options.
            # Qwen2.5_7B，Qwen2.5_14B，Qwen2.5_32B，Qwen2.5_72B，Qwen2_7B，Qwen2_57B，Qwen2_72B，Llama3.3_70B，glm_4_9B_chat
            payload = json.dumps({
                'text': prompt,
                "model_name":"Qwen2_57B",
                "temperature":0,
                "max_tokens":20
                })
            # Send a request and get the response.
            response = requests.request("POST", url, headers=headers, data=payload,timeout=30)
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

print("\nModel call started\n")
cnt_irr=0

# Process each raw citation data, where each raw citation data is a dictionary with three fields: category, label_prompt, and output.
# Return each raw citation data in a dictionary, including the original category, prompt, output, along with the model's response.
def item_processing(dic:dict):
    prompt = dic['prompt']
    try:
        # Call the citation_generation function to get a new response.
        new_response = citation_generation(prompt)['response']
        end_index = new_response.find(']')
        new_response = new_response[0:end_index]
        # Modify the value of the 'response' key in the original dictionary.
        dic['response'] = new_response
    except (TypeError, KeyError) as e:
        dic['response'] = "{}: {}".format(type(e).__name__, e)

    return dic
           
def process_list_and_write_to_file(data_list: list, output_file: str):
    # Create a new list to store the final results.
    num = 0 
    handling_error_num=0
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("[")
        f.close()
    for item in tqdm(data_list, desc="Processing items"):
        if "RunTimeError Message\n\n" not in item.get("response"):
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

process_list_and_write_to_file(data_citation_combo, output_file)
