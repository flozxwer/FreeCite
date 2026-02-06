import json
import re
from tqdm import tqdm
import sys

model = "Qwen2.5_32B"
f_res='' # filename of data
log_file_path = "" # filename of output log

sys.stdout = open(log_file_path, 'w', encoding='utf-8')
print("result file name:",f_res)

with open(f_res, 'r', encoding='utf-8') as file:
    content=file.read()
if content.endswith(',]'):
    content=content[:-2]+']'
if content.endswith(','):
    print('end with comma')
    content=content[:-1]+']'
qwen25_res = json.loads(content)
print(model, 'Result count:',len(qwen25_res))
prompts = []
for it in qwen25_res:
    prompts.append(it['prompt'])
print(model, 'unique prompt count:',len(set(prompts)))

# Load reference file
ref_path = '../3000_sample.json'
with open(ref_path, 'r', encoding='utf-8') as file:
    ref_data = json.load(file)
# Check that the model result file is consistent with the reference file
cnt_match=0
for ind in range(0,len(ref_data)):
    output1 = qwen25_res[ind]['output']
    output2 = ref_data[ind]['output']
    if output1 != output2:
        print('error, output mismatch')
    else:
        cnt_match=cnt_match+1
if cnt_match==len(ref_data):
    print('The model results match the original data')
else:
    print('Model result does not match raw data, there are errors ',ref_data-cnt_match,' place ')
print('-----------------')
print('Irregular output：')

# get right answer
def get_right_answer(res:dict, raw_answer: str):

    # :res: Model output
    # :param raw_answer: Complete original answer
    # :return: The correct answer candidate for this position

    #div_string="<user_end><im_assistant>" # Closed-source model
    div_string = "<|im_end|>\n<|im_start|>assistant\n"  # qwen
    #div_string = "\n\nAssistant:"  # deepseek
    # div_string = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"  # Llama
    #div_string="<|assistant|>\n" # glm
    
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
for index in range(0,len(ref_data)):
    ref_item = ref_data[index]
    res = qwen25_res[index]
    cnt+=1
    dic={}
    category = res.get('category')
    output = ref_item.get('output') # Reference files, raw data can tell us what the standard answer is
    prompt = ref_item.get('prompt') # Reference files, raw data can tell us what the standard answer is
    response = res.get('response')

    if "Error:" in response or "RunTimeError" in response or "当前分组上游负载已饱和" in response:
        cnt-=1
        continue

    response='['+response+']'

    correct_answer=get_right_answer(res=ref_item,raw_answer=output) # Raw data tells us the standard answer
    
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
print(model, 'Models cite correct numbers:', corr)  # Output the correct number of citations for the model
print(model, 'Correct rate of model citation:', round(corr / cnt * 100, 2)) 

new_outputs=[it['output'] for it in new_res]
print('The number of outputs after removing irregular data:',len(set(new_outputs)))

sys.stdout.close()
