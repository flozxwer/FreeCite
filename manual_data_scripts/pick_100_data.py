# Prepare to generate prompts for manual annotation data.

import json
import re
from tqdm import tqdm
import requests
import time
import random
from collections import defaultdict
random.seed(30)

# Load reference data
fname='' # filename of data
with open(fname, 'r', encoding='utf-8') as file:
    data_citation_combo = json.load(file)
prompts=[item['prompt'] for item in data_citation_combo]
print('prompts count:',len(prompts))
print('unique prompts count:',len(set(prompts)))
print('/....../....../....../....../....../')

# Input the concatenated version of the prompt, output the prompt for manual annotation.
def generate_label_prompt(prompt_ori):
    # Remove the prompt after 'assistant' from the prompt.
    div_string='<|im_end|>\n<|im_start|>assistant\n'
    label_prompt = prompt_ori.partition(div_string)[0] + div_string

    return label_prompt

# Remove duplicates, randomly select 100 outputs.
dic_mapping={}
for item in data_citation_combo:
    output=item['output']
    if output not in dic_mapping.keys():
        dic_mapping[output]=[item]
    else:
        dic_mapping[output].append(item)

print('Number of chapters after processing:',len(dic_mapping.keys()))
# Randomly select 100 articles.
outputs=list(dic_mapping.keys())
sampled_outputs = random.sample(outputs,100)
print('Number of uniquely selected chapters:',len(set(sampled_outputs)))
# Generate 100 unique chapters.
lis_raw_label_data=[]  # Original data for annotation
for output_key in sampled_outputs:
    ite = dic_mapping[output_key]
    prompt=ite[0]['prompt']
    label_prompt=generate_label_prompt(prompt)
    category=ite[0]['category']
    dic = {
        "category":category,
        "output":output_key,
        "label_prompt":label_prompt
    }
    lis_raw_label_data.append(dic)

fname=''  # raw label data
with open(fname, 'w', encoding='utf-8') as file:
    json.dump(lis_raw_label_data,file,indent=4,ensure_ascii=False)
