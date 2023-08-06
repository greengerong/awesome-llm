
from datasets import load_dataset

data_files = ['./dataset/leetcode-solutions.jsonl']

leetcode_dataset = load_dataset("json", data_files=data_files)
# print(leetcode_dataset)
# print(leetcode_dataset['train'].shuffle(seed=42).select(range(2))[0])

def parseLang(item, lang):
    answer = item['answer'][lang]
    explanation = item['answer']['explanation']
    return f'''
    {answer}    
    {explanation}
    '''

new_leetcode_dataset = leetcode_dataset.map(lambda  item: {'id': int(item['id'])})

new_leetcode_dataset = new_leetcode_dataset.map(lambda  item: {
        'java': parseLang(item, 'java'),
        'c++': parseLang(item, 'c++'),
        'python': parseLang(item, 'python'),
        'javascript': parseLang(item, 'javascript')
    },  remove_columns=['answer'])

# print(new_leetcode_dataset['train'][0])

for split, dataset in new_leetcode_dataset.items():
    dataset.to_json(f"./output/leetcode-{split}.jsonl")

new_leetcode_dataset.set_format("pandas")
pd =  new_leetcode_dataset['train'][:]

pd.to_json(f"./output/leetcode.json", orient='records')

new_leetcode_dataset.reset_format()