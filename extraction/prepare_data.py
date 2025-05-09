import random
from copy import deepcopy
from schemas import process_templates
import json
import numpy as np

system = "You are an expert on zeolite materials who extracts structured synthesis data from experimental procedures described in research papers."

def construct_user_prompt(name, source, processes, text):
    prompt = f"""
You are provided with a text passage describing the synthesis of zeolite materials from a research article alongside a zeolite sample name, the source of the original zeolite and a list of any post-synthetic modifications.
Your task is to extract the key synthesis or commercial information from the source and each of the post-synthetic modifications in structured JSON format.
If you do not believe a step in the post-synthetic processes applies to this sample, you should return "did step": "false" in the dictionary for that step.
Only include the information that is mentioned in the text. Do not hallucinate.

Sample: {name}

Original Zeolite Source: {source}

Post-synthesis processes: {', '.join(processes)}

Text: {text}
"""
    return prompt.strip()

def construct_assistant_prompt(sample):
    output = {
        "zeolite_source": sample['zeolite_source'],
        "post_synthesis": [step for step in sample['post-synthesis']]
    }
    return json.dumps(output, indent=2)

def clean_calcination_steps(data):
    for paper in data:
        for sample in paper.get("samples", []):
            for step in sample.get("post-synthesis", []):
                if step.get("process") == "calcination":
                    step.pop("start_form", None)
                    step.pop("end_form", None)
    return data

def select_one_sample_per_history_length(paper, biases):
    """
    From one paper's samples, select one sample for each unique history length - only used to prepare training data.
    """
    selected = []
    unselected = []
    grouped = {}

    if paper['doi'] in biases:
        return paper.get("samples", []), []

    for sample in paper.get("samples", []):
        post_synth = sample.get("post-synthesis", [])
        source = sample.get("zeolite_source", {}).get("source", "")
        if reject_sample(source, post_synth)[0]:
            continue
        post_synthesis_len = len(sample.get("post-synthesis", []))
        grouped.setdefault(post_synthesis_len, []).append(sample)

    for post_synthesis_len, samples in grouped.items():
        chosen_sample = random.choice(samples)
        selected.append(chosen_sample)
        samples.remove(chosen_sample)
        unselected.extend(samples)

    return selected, unselected

def reject_sample(source, processes):
    supported_sources = {"commercial", "hydrothermal crystallization"}
    supported_processes = {"calcination", "ion exchange", "hydrothermal crystallization", "chemical liquid deposition", "steam treatment", "solvent etching", "recrystallization"}

    if source.replace('_', " ").strip().lower() not in supported_sources:
        return True, "unknown source"

    if len(processes) == 0:
        return False, "accepted"

    for process in processes:
        if process['process'].replace('_', " ").strip().lower() not in supported_processes:
            return True, f"unsupported post-synthesis: {process}"
        
    return False, "accepted"


def generate_fake_step(process_name, type):
    """Generate a fake step for a given process name."""
    step = deepcopy(process_templates.get(process_name, {}))
    if type == 'process':
        step["process"] = process_name 
        step['did_step'] = "false"
    elif type == 'source':
        step["source"] = process_name
        step['did_step'] = "false"
    else:
        raise ValueError(f"Unknown type: {type}")
    return step


def inject_fake_steps(data, fake_process_rate, swap_source_rate):
    modified_data = deepcopy(data)
    all_samples = []
    for paper in modified_data:
        for sample in paper.get("samples", []):
            all_samples.append((paper, sample))
            rand_process = np.random.rand()
            rand_swap = np.random.rand()
            if rand_process < fake_process_rate:
                used_processes = [step.get('process') for step in sample.get('post-synthesis', []) if 'process' in step]
                candidate_processes = [
                    p for p in process_templates.keys()
                    if p not in used_processes and p != "commercial" and p != "hydrothermal crystallization"
                ]
                if not candidate_processes:
                    continue  # skip if no unused process

                selected_fake_process = random.choice(candidate_processes)
                fake = generate_fake_step(selected_fake_process, "process")
                insert_at = random.randint(1, len(sample['post-synthesis'])) if sample['post-synthesis'] else 0
                sample['post-synthesis'].insert(insert_at, fake)

            if rand_swap < swap_source_rate:
                if sample['zeolite_source']['source'] == "commercial":
                    sample['zeolite_source'] = generate_fake_step("hydrothermal crystallization", "source")
                elif sample['zeolite_source']['source'] == "hydrothermal crystallization":
                    sample['zeolite_source'] = generate_fake_step("commercial", "source")
                else: 
                    selected_fake_source = random.choice(["commercial", "hydrothermal crystallization"])
                    sample['zeolite_source'] = generate_fake_step(selected_fake_source, "source")
    return modified_data



def prepare_train_and_test_data(datafile, outfile_train, outfile_test, fake_process_rate=0.25, swap_source_rate=0.25):
    # only 1 sample per unique history length from each paper are selected - the rest go into a provisional test dataset
    # all samples in biases are used for training - useful if the paper contains processes that are not very common
    biases = ["10.1016/j.ces.2024.120763"]
    train = []
    test = []
    with open(datafile, 'r') as file:
        data = json.load(file)

    data = clean_calcination_steps(data)
    data_with_fakes = inject_fake_steps(data, fake_process_rate, swap_source_rate) 
    for paper in data_with_fakes:
        selected_samples, unselected_samples = select_one_sample_per_history_length(paper, biases)

        for sample in selected_samples:
            user = construct_user_prompt(
                name=sample['name'],
                source=sample['zeolite_source']['source'],
                processes=[step['process'] for step in sample['post-synthesis']],
                text=paper['text']
            )
            assistant = construct_assistant_prompt(sample)
            new = {
                'doi': paper['doi'],
                'sample': sample['name'],
                'messages': [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': user},
                    {'role': 'assistant', 'content': assistant}
                ]
            }
            train.append(new)

        for sample in unselected_samples:
            user = construct_user_prompt(
                name=sample['name'],
                source=sample['zeolite_source']['source'],
                processes=[step['process'] for step in sample['post-synthesis']],
                text=paper['text']
            )
            assistant = construct_assistant_prompt(sample)
            new = {
                'doi': paper['doi'],
                'sample': sample['name'],
                'messages': [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': user},
                    {'role': 'assistant', 'content': assistant}
                ]
            }
            test.append(new)
    with open(outfile_train, 'w') as f_train:
        for example in train:
            f_train.write(json.dumps(example) + '\n')

    with open(outfile_test, 'w') as f_test:
        for example in test:
            f_test.write(json.dumps(example) + '\n')


if __name__ == "__main__":
    datafile = '../data/manual_extraction_cleaned.json'
    outfile_train = 'extraction_train.jsonl'
    outfile_test = 'extraction_test.jsonl'
    prepare_train_and_test_data(datafile, outfile_train, outfile_test, fake_process_rate=0.25, swap_source_rate=0.25)