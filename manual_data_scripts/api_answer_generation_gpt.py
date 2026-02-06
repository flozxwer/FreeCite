import json
from tqdm import tqdm
import requests
import concurrent.futures
import threading
from tqdm import tqdm

key = ""
# gpt-4-turbo    gpt-4o    gpt-4o-mini 
model = "gpt-4o"

fname='' # filename of data
with open(fname, 'r', encoding='utf-8') as file:
    prompt_for_artificial_data=json.load(file)
print(len(prompt_for_artificial_data))

def chat_with_gpt(query: str,
                  key: str = key,
                  model: str = model,
                  system_message: str = None,
                  temperature: float = 0,
                  retry_time: int = 5,
                  json_mode: bool = False
                  ):
    url = ""
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
            #print(response.text)
            result = json.loads(response.text)["choices"][0]["message"]["content"]
            break
        except Exception as e:
            count = count + 1
            print(e)
            if count > retry_time:
                raise Exception('ReturnCode.LLM_ERROR')
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
        ans_raw = chat_with_gpt(new_prompt)
    except (TypeError, KeyError) as e:
        return {
            "category": category,
            "output": output,
            "gpt prompt": new_prompt,
            "response": "{}: {}".format(type(e).__name__, e)
        }
    dic_new={
        'category': category,
        'gpt prompt': new_prompt,
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

# For debugging
def parallel_processing_try(items):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for result in tqdm(executor.map(item_processing, items), total=len(items)):
            with lock:
                print(result)

parallel_processing(prompt_for_artificial_data)
