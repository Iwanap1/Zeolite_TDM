import json

def prepare_process_identifation_data(data):
    system = "you are a zeolite expert. You will be given a list of zeolite names and a text. Your task is to identify the processes and original zeolite source involved in the synthesis or modification of a zeolite"
    new = []
    for paper in data:
        zeolite_names = [sample['name'] for sample in paper['samples']]
        new_paper = {
            'doi': paper['doi'],
            'messages': [{'role': 'system', 'content': system}, 
                        {'role': 'user', 'content': construct_process_identification_user_prompt(zeolite_names, paper['text'])},
                        {'role': 'assistant', 'content': construct_process_identification_assistant_prompt(paper['samples'])}]
        }
        new.append(new_paper)

    with open('process_identification_data.jsonl', 'w') as file:
        for paper in new:
            file.write(json.dumps(paper) + '\n')


def construct_process_identification_user_prompt(zeolite_names, text):
    prompt = f"""
    Use the provided passage to identify the morphological description, original zeolite source, and post-synthesis processes involved in the synthesis of the following zeolites:
    You may choose from the following zeolite sources: commercial, hydrothermal crystallization or unknown
    You may choose from the following post-synthetic processes: calcination, ion exchange, solution etching, chemical liquid deposition, recrystallization, steam treatment, impregnation or something else
    Consecutive acid and alkaline treatments are considered different solvent etching processes.

    Zeolites: {', '.join(zeolite_names)}

    Passage: {text}

    Please provide the information in JSON format without extra commentrary. 
"""
    return prompt


def construct_process_identification_assistant_prompt(samples):
    output = {}
    for sample in samples:
        output[sample['name']] = {
            'morphological_description': sample['morphological_description'],
            'zeolite_source': sample['zeolite_source']['source'],
            'post_synthesis': [step['process'] for step in sample['post-synthesis']]
        }
    return json.dumps(output, indent=2)


# def prepare_extraction_data(data):
#     return

if __name__ == '__main__':
    data_file = "../manual_extraction.json"
    with open(data_file, 'r') as file:
        data = json.load(file)
    prepare_process_identifation_data(data)
