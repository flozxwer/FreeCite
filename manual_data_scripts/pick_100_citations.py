import re
import json

# Determine whether the end of a string contains a citation ID in the form of [abcd1234].
def has_reference_id_at_end(text):
    # Regular expression: Match the format [abcd1234] at the end.
    pattern = r"\[[a-zA-Z0-9]{8}\]$"
    return bool(re.search(pattern, text))

fname='' # filename of data
with open(fname, 'r', encoding='utf-8') as file:
    content=file.read()
if content.endswith(',]'):
    content=content[:-2]+']'
artificial_data = json.loads(content)
print('Loaded filename:',fname)
# Remove strange sentences from the output
def remove_irregular_statements(response,sentences):
    pattern = r'#+\s*\n+\s*参考资料：\s*'
    pattern2 = r'\n+\s*参考资料：\s*'
    pattern3 = r'\n+\s*\[参考资料：\s*'
    pattern4 = r'#+\s*参考资料\s*'

    output = response
        
    # Find all matching substrings
    lis_id = re.findall(pattern, output)
    lis_id2 = re.findall(pattern2, output)
    lis_id3 = re.findall(pattern3, output)
    lis_id4 = re.findall(pattern4, output)
        
    # Merge all matched substrings into a list, provided that each list is not empty.
    target_strs = []
    if lis_id:
        target_strs.extend(lis_id)
    if lis_id2:
        target_strs.extend(lis_id2)
    if lis_id3:
        target_strs.extend(lis_id3)
    if lis_id4:
        target_strs.extend(lis_id4)
           
    # Remove sentences that contain the target substring.
    if target_strs:
        sentences_to_remove = set()
        for i, sentence in enumerate(sentences):
            for target_str in target_strs:
                if target_str in sentence:
                    sentences_to_remove.add(i)
                    break
        # Remove sentences based on index.
        filtered_sentences = [sentence for i, sentence in enumerate(sentences) if i not in sentences_to_remove]  
    else:
         filtered_sentences= sentences 
    return  filtered_sentences
citation_num_bar=100
#total_citation_cnt=0

pattern_2=r'\[[a-zA-Z0-9]{8}\]'
pattern = r'\[[a-zA-Z0-9]{8}\].+\[[a-zA-Z0-9]{8}\]'
# Return the sentences, prompts, and categories corresponding to the first 100 citations in the data dataset.
# The sentence is the key, and the prompt and category are within the value corresponding to the sentence key.
# The data dataset is a list of dictionaries, each containing a prompt and a response.
# Three columns of category
def raw_label_data(data):
    final_sentences={} # The final returned data
    total_citation_cnt=0 # Record the number of citations
    for item in data:
        category_here=item.get('category', '')
        prompt_here=item.get('prompt', '')
        response = item.get('response', '')
        #print(response)
        response = re.sub(r'#######\n\[回答\]: # |\n########|#############\n\[综述\]:|#######\n\[回答\]:|\[综述\]:|\[回答\]: ', '', response)
        sentences = [s.strip() for s in response.split('。') if s.strip()]
        filtered_sentences=remove_irregular_statements(response,sentences)
        lis_id=re.findall(pattern_2,filtered_sentences[-1])
        if lis_id:
            filtered_sentences=filtered_sentences[:-1]
        for index in range(len(filtered_sentences)):
            sentence=filtered_sentences[index]
            if sentence in final_sentences.keys():
                continue
            sentence_for_test= re.sub(r'\[.*?\]|\[.*', '', sentence)
            if len(sentence_for_test)<30 and index >0:
                sentence=filtered_sentences[index-1]+'。'+sentence
                print('add sentence')
            matches = re.findall(pattern, sentence)
            if matches:
                continue
            if has_reference_id_at_end(sentence):
                num_citations=len(re.findall(pattern_2,sentence))
                total_citation_cnt=total_citation_cnt+num_citations
                
                final_sentences[sentence]={
                    "prompt":prompt_here,
                    "category":category_here
                }
            if total_citation_cnt>=100:
                print(total_citation_cnt)
                return final_sentences
    return final_sentences

final_sentences_dic = raw_label_data(artificial_data)
#print(final_sentences_dic.keys())
cnt_here=0
for ite in final_sentences_dic.keys():
    print("\n",ite)
    cnt_here=cnt_here+len(re.findall(pattern_2,ite))
print('The number of verified citations:',cnt_here)

# Process the final citation
pattern_sep = re.compile(r'(\[[a-zA-Z0-9]{8}\])(.*?)(?=\[[a-zA-Z0-9]{8}\]|\Z)', re.DOTALL) # 引证id和对应引证资料

processed_data = []
for key in final_sentences_dic.keys():
    data=final_sentences_dic[key]
    category=data['category']
    prompt=data['prompt']
    answer=key
    if "你是一个中文大语言模型。你在做一个百科问答任务，" in prompt:
        chunk_ref=prompt.partition("\n\n参考资料：\n")[-1].partition("相关问答：")[0].partition("提示思路：")[0].partition("\n\n\n结构化模版：\n")[0]
    else:
        chunk_ref=prompt.partition("\n\n参考资料：\n")[-1].partition("注意遵守以下事项：\n1. 你需要在回答结果中插入引用证据的来源编号，格式为[编号]")[0]
    ids=re.findall(pattern_2,chunk_ref)
    
    all_refs = pattern_sep.findall(chunk_ref)
    if len(ids)!=len(all_refs):
        print('error, number mismatch')
    # Save to the dictionary
    refs = {}
    for ref_id, content in all_refs:
        refs[ref_id] = content
    # Highlight the citations in the answer one by one in red.
    matches=re.finditer(pattern_2,answer)
    cnt=0
    for match in matches:
        #print(match.group(),'   ',match.start(),'   ',match.end())
        id=match.group()
        start_ind=match.start()
        end_ind=match.end()
        before_citation_id=answer[:start_ind]
        new_id="<span style='color:red'>{}</span>".format(id)
        answer_for_label=before_citation_id+new_id # Highlight the citations and all the preceding text in red.

        # Search for the reference corresponding to the id in the prompt.
        if id in refs:
            ref = refs[id]
            id_new="<span style='color:red'>{}</span>".format(id)
            ref_for_label=id_new+ref
        else:
            ref_for_label="无对应参考资料序号"
            print(f"{id}Reference not found.")
            #print(chunk_ref)
        cnt=cnt+1

        ref_html = ref_for_label.replace("\n", "<br/>")
        answer_html = answer_for_label.replace("\n", "<br/>")

        processed_data.append({
            'category': category,
            'reference': ref_html,
            'answer': answer_html
        })

print('Example of single sentence and single citation matching data:',len(processed_data))

for item in processed_data[-5:]:
    print(item['reference'])
    print(item['answer'])
    
file_name='' # filename of processed data
with open(file_name, "w", encoding="utf-8") as file:
    json.dump(processed_data, file, indent=4, ensure_ascii=False)
