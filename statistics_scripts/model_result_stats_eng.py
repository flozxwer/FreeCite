import json
import re
from tqdm import tqdm
import sys
import difflib

model = "gemini"
f_res='/../*.json'  # filename of data
log_file_path = "/../*_log.txt"  # filename of output log
sys.stdout = open(log_file_path, 'w', encoding='utf-8')

model_div_string_dict={
    "qwen2.5_72B": "<|im_end|>\n<|im_start|>assistant\n",
    "qwen2.5_32B": "<|im_end|>\n<|im_start|>assistant\n",
    "qwen2.5_14B": "<|im_end|>\n<|im_start|>assistant\n",
    "qwen2.5_7B":"<|im_end|>\n<|im_start|>assistant\n",
    "qwen2_72B": "<|im_end|>\n<|im_start|>assistant\n",
    "qwen2_57B": "<|im_end|>\n<|im_start|>assistant\n",
    "qwen2_7B": "<|im_end|>\n<|im_start|>assistant\n",
    "baichuan": "<user_end><im_assistant>",
    "doubao": "形式如'[abcd1234]'：\n", # doubao_oldprompt
    "moonshot": "<user_end><im_assistant>",
    "glm_4_plus": "<user_end><im_assistant>",
    "deepseek_v3": "<user_end><im_assistant>",
    "glm_4_9b_chat": "<|assistant|>\n",
    "llama3.3_70B": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "gpt_4_turbo": "<user_end><im_assistant>",
    "gpt_4o": "<user_end><im_assistant>",
    "gpt_4o_mini": "<user_end><im_assistant>",
    "qwen_api": "<user_end><im_assistant>",
    "claude":"<user_end><im_assistant>",
    "gemini":"<|im_end|>\n<|im_start|>assistant\n"
}

with open(f_res, 'r', encoding='utf-8') as file:
    content=file.read()
if content.endswith(',]'):
    content=content[:-2]+']'
if content.endswith(','):
    print('end with comma')
    content=content[:-1]+']'
data_res = json.loads(content)
print(model, 'Result count:',len(data_res))
prompts = []
for it in data_res:
    prompts.append(it['prompt'])
print(model, 'unique prompt count:',len(set(prompts)))
print('-----------------')
print('Irregular output：')

# get right answer
def get_right_answer(res:dict, raw_answer: str):

    # :res: Model output
    # :param raw_answer: Complete original answer
    # :return: The correct answer candidate for this position

    #div_string="<user_end><im_assistant>" # Closed-source model
    #div_string="形式如'[abcd1234]'：\n" # doubao——oldprompt
    #div_string = "<|im_end|>\n<|im_start|>assistant\n"  # qwen
    #div_string = "\n\nAssistant:"  # deepseek
    #div_string = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"  # Llama
    #div_string="<|assistant|>\n" # glm
    div_string = model_div_string_dict[model]
    
    prefix = res.get('prompt').partition(div_string)[-1][:-1] # delete"["
    answer_after_prefix = raw_answer.partition(prefix)[-1]
    answer_candidates = []
    if prefix not in raw_answer:
        print('Incorrect match, no sentence in the original text')
        print("sentence: ",prefix)
        return []
    while True:
        # Matches the citation for the first shape like [abcd1234] in the text anser_after_prefix
        candidate = re.match(r"\[([a-zA-Z0-9]{8})\]", answer_after_prefix)
        # If there is one, remove that quote and match it to the next.
        if candidate:
            answer_candidates.append(candidate.group())
            answer_after_prefix = answer_after_prefix.partition(candidate.group())[-1]
        else:
            break
    return answer_candidates

cnt=0
irrg=0
corr=0
new_res = []
for res in data_res:
    cnt+=1
    dic={}
    category = res.get('category')
    output = res.get('output')
    prompt = res.get('prompt')
    response = res.get('response')
    eng_prompt = res.get('eng prompt')

    if "Error:" in response or "RunTimeError" in response or "当前分组上游负载已饱和" in response:
        cnt-=1
        continue
    # for claude
    """ 
    div_string = model_div_string_dict['claude']
    prompt_suffix = eng_prompt.partition(div_string)[-1]

    last_bracket_pos = prompt_suffix.rfind('[')
    if last_bracket_pos == -1:
        cnt -= 1
        continue

    text_before_bracket = prompt_suffix[:last_bracket_pos]
    cn_dot_pos = text_before_bracket.rfind('.')

    if cn_dot_pos == -1:
        prefix_in_prompt = prompt_suffix[:last_bracket_pos + 1]
    else:
        prefix_in_prompt = prompt_suffix[cn_dot_pos + 1: last_bracket_pos + 1]

    n = prefix_in_prompt.count('[')

    threshold = 0.80
    match = difflib.get_close_matches(
        prefix_in_prompt,
        [response[i:i+len(prefix_in_prompt)] for i in range(len(response)-len(prefix_in_prompt)+1)],
        n=1,
        cutoff=threshold
    )
    if not match:
        cnt -= 1
        continue

    idx = response.find(match[0])

    bracket_count = 0
    start = idx
    for i in range(idx, len(response)):
        if response[i] == '[':
            bracket_count += 1
            if bracket_count == n:
                start = i + 1
                break

    candidate = response[start: start + 8]
    response = f'[{candidate}]'
    print(response) """

    # original ver.
    if ']' in response:
        index = response.find(']')
        response = response[max(0, index - 8):index]

    response='['+response+']'
    correct_answer=get_right_answer(res=res,raw_answer=output)
    
    if not correct_answer:
        print('error! no right choice!')
        print('no right choice///',response)
        cnt-=1
        continue
    if 'Cite-' in output or 'Cite-' in prompt:
        print('irregular input!//')
        cnt-=1
        continue
    if response is None:
        print('err in final response')
    
    else:
        if len(response) !=10:
            irrg+=1
            print("response: ", response)
            print("     correct answer: ", correct_answer)
            #continue # Delete the wrong data
        dic['category']=category
        dic['output']=output
        dic['prompt']=prompt
        dic['eng prompt']=eng_prompt
        dic['response']=response
        dic['correct answer']=correct_answer

        if response in correct_answer:
            corr+=1
            dic['correctness']=True
        else:
            dic['correctness']=False

        new_res.append(dic)

print('-----------------')
print('citation count:', cnt) # Number of entries processed
print('Irregular data number:', irrg) # Number of problematic entries
print(model, 'model cite correct numbers:', corr)  # Output the correct number of citations for the model
print(model, 'correct rate of model citation:', round(corr / cnt * 100, 2)) 

new_outputs=[it['output'] for it in new_res]
print('The number of outputs after removing irregular data:',len(set(new_outputs)))

sys.stdout.close()
