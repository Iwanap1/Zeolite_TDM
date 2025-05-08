import random
from copy import deepcopy
from schemas import process_templates


def select_one_sample_per_history_length(paper, biases):
    """
    From one paper's samples, select one sample for each unique history length.
    """
    selected = []
    grouped = {}

    if paper['doi'] in biases:
        return paper.get("samples", [])

    for sample in paper.get("samples", []):
        history_len = len(sample.get("history", []))
        grouped.setdefault(history_len, []).append(sample)

    for history_len, samples in grouped.items():
        chosen_sample = random.choice(samples)
        selected.append(chosen_sample)

    return selected


def generate_fake_step(process_name, type):
    """Generate a fake step for a given process name."""
    step = process_templates.get(process_name, {})
    if type == 'process':
        step["process"] = process_name 
        step['did step'] = "false"
    elif type == 'source':
        step["source"] = process_name
        step['did step'] = "false"
    else:
        raise ValueError(f"Unknown type: {type}")
    return step


def inject_fake_steps(data, fake_process_rate=0.25, swap_source_rate=0.25):
    modified_data = deepcopy(data)

    all_samples = []
    for paper in modified_data:
        for sample in paper.get("samples", []):
            all_samples.append((paper, sample))



def filter_samples(process_identification_output):
    # removes impregnated samples and samples with no or unknown history/source
    return