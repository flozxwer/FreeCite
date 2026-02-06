import json
import random
import os

'''先分别从五个文件里随机挑选20条数据，保存成5个文件'''
# with open('/path/to/result/eli5-baichuan4-turbo-bm25-shot2-ndoc20-42.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# new_data = data.copy()
# if len(new_data['data']) > 20:
#     new_data['data'] = random.sample(new_data['data'], 20)

# with open('/path/to/result/new/random_20_data_bm25.json', 'w', encoding='utf-8') as f:
#     json.dump(new_data, f, indent=4, ensure_ascii=False)

# print("Saved 20 random items to random_20_data.json")

'''如果是另一个模型的文件，将其分别与之前挑选的20条数据进行比对'''
def load_sample_ids(sample_file):
    # 加载挑选的20条数据
    with open(sample_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Extract unique identifiers - using question as the key
    sample_ids = set()
    for item in data['data']:
        question = item['question']
        sample_ids.add(question)
    return sample_ids

def filter_matching_items(source_file, sample_ids):
    # Filter items from source file that match the sample_ids
    with open(source_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    matching_data = []
    for item in data['data']:
        question = item['question']
        if question in sample_ids:
            matching_data.append(item)
    
    # Create new data structure with only matching items
    # new_data = {
    #     "args": data["args"],
    #     "data": matching_data
    # }
    # return new_data
    return matching_data

def get_source_file(model, type):
    # 根据type确定数据集和ndoc参数
    if type in ['asqa_oracle', 'dpr', 'gtr']:
        dataset = 'asqa'
    else:
        dataset = 'eli5'
    
    # 确定ndoc参数
    if type in ['oracle', 'asqa_oracle', 'eli5_oracle']:
        ndoc = '5'
    else:
        ndoc = '20'
    
    # 特殊处理type名称在文件名中的表示
    file_type = type.replace('asqa_', '').replace('eli5_', '')
    
    return f"/path/to/result/{dataset}-{model}-{file_type}-shot2-ndoc{ndoc}-42.json"

def main():
    # 需要更改的变量
    #models = ['baichuan4-turbo', 'deepseek-v3', 'ep-20250123113406-7vzr5', 'glm_4_9B_chat', 'Llama3.3_70B_100', 'moonshot-v1-32k', 'Qwen2_7B', 'Qwen2_57B', 'Qwen2_72B', 'Qwen2.5_7B', 'Qwen2.5_14B', 'Qwen2.5_32B', 'Qwen2.5_72B']
    models = ['gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini']
    types = ['asqa_oracle', 'dpr', 'gtr', 'eli5_oracle', 'bm25']
    #type = 'eli5_oracle' # asqa_oracle、bm25、dpr、eli5_oracle、gtr
    # # 待比对的数据
    #source_file = "/path/to/result/eli5-deepseek-v3-oracle-shot2-ndoc5-42.json"
    # # 挑选出的20条数据
    # sample_file_format = "/path/to/result/new/random_20_data_type.json"
    # sample_file = sample_file_format.replace('type', type)
    # # 输出文件
    # output_file_format = "/path/to/result/new/model_type_20_data.json"
    # output_file = output_file_format.replace('model', model).replace('type', type)
    # #print(output_file)

    # sample_ids = load_sample_ids(sample_file)
    # matching_data = filter_matching_items(source_file, sample_ids)

    for model in models:
        final_data = {
            "model": model,
            "data": []
        }

        for type in types:
            # 待比对的数据
            source_file = get_source_file(model, type)
            
            # 挑选出的20条数据
            sample_file = f"/path/to/result/new/random_20_data_{type}.json"
            
            sample_ids = load_sample_ids(sample_file)
            try:
                sample_ids = load_sample_ids(sample_file)
                matching_data = filter_matching_items(source_file, sample_ids)
                
                # 将匹配的数据添加到最终结果中
                final_data['data'].extend(matching_data)
                print(f"Processed {type}: found {len(matching_data)} matches")
            except Exception as e:
                print(f"Error processing {type}: {str(e)}")
                continue

        # 输出文件
        output_file = f"/path/to/result/100data1/{model}_100_data.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)
        
        print(f"\nSaved total {len(final_data['data'])} matching items to {output_file}\n")

if __name__ == "__main__":
    main()

