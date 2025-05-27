import json
import pandas as pd

files = [
    "test_llama3_flat_table.json", "test_llama3_string.json", "test_llama3_table.json"
]

# Define all possible fields
fields = [
    "sample", "Si_Al", "V_total", "V_micro", "V_meso", "Bronsted_Acid_Sites",
    "Lewis_Acid_Sites", "S_BET", "S_ext", "D_micro", "D_meso", "crystallinity", "metal_content"
]

with open('test_data.json', 'r') as f:
    data = json.load(f)
samples = []
for paper in data:
    for sample in paper['samples']:
        sample_data = {field: sample.get(field, "") for field in fields}
        sample_data['doi'] = paper['doi']
        sample_data['sample'] = sample['sample']
        sample_data['source'] = 'real'
        sample_data.update(sample.get('data', {}))
        samples.append(sample_data)
df = pd.DataFrame(samples)

all_samples = df['sample'].unique().tolist()
samples = []
for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
    for paper in data:
        for sample in all_samples:
            pred_samples = paper.get('pred', [])
            if sample in [s['sample'] for s in pred_samples]:
                pred_sample = pred_samples[[s['sample'] for s in pred_samples].index(sample)]
                sample_data = {field: pred_sample.get(field, "") for field in fields}
                sample_data['doi'] = paper['doi']
                sample_data['sample'] = sample
                sample_data['source'] = file.replace('.json', '')
                samples.append(sample_data)

pred_df = pd.DataFrame(samples)
all_df = pd.concat([df, pred_df], ignore_index=True)
all_df.sort_values(by=['doi', 'sample'], inplace=True)
all_df.to_csv('test_results.csv', index=False)