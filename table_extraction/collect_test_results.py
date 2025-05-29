import json
import pandas as pd

# Input files
real_data_file = "test_data.json"
prediction_files = [
    "test_llama3_flat_table.json",
    "test_llama3_string.json",
    "test_llama3_table.json"
]

# Define all possible fields
fields = [
    "sample", "Si_Al", "V_total", "V_micro", "V_meso", "Bronsted_Acid_Sites",
    "Lewis_Acid_Sites", "S_BET", "S_ext", "D_micro", "D_meso", "crystallinity", "metal_content"
]

# Load real data and assign consistent table numbers
with open(real_data_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

samples = []
table_keys = {}
table_counter = 1

for paper in data:
    sample_list = tuple(s['sample'] for s in paper['samples'])
    key = (paper['doi'], sample_list)

    if key not in table_keys:
        table_keys[key] = table_counter
        table_counter += 1

    table_number = table_keys[key]

    for sample in paper['samples']:
        sample_data = {field: sample.get(field, "") for field in fields}
        sample_data['doi'] = paper['doi']
        sample_data['sample'] = sample['sample']
        sample_data['source'] = 'real'
        sample_data['table_number'] = table_number
        sample_data.update(sample.get('data', {}))  # In case of nested fields
        samples.append(sample_data)

df = pd.DataFrame(samples)

# Process predictions
samples = []
for file in prediction_files:
    with open(file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    for table_index, paper in enumerate(pred_data):
        sample_list = tuple(s['sample'] for s in paper.get('real', []))
        key = (paper['doi'], table_index, sample_list)

        if key not in table_keys:
            table_keys[key] = table_counter
            table_counter += 1

        table_number = table_keys[key]
        file_label = file.replace('.json', '').replace('test_llama3_', '')

        # Get only real sample names from this specific table
        real_samples_in_this_table = [s['sample'] for s in paper.get('real', [])]

        for sample_name in real_samples_in_this_table:
            pred_samples = paper.get('pred', [])
            pred_sample = next((s for s in pred_samples if s['sample'] == sample_name), None)

            if pred_sample:
                sample_data = {field: pred_sample.get(field, "") for field in fields}
            else:
                sample_data = {field: "" for field in fields}

            sample_data['table_number'] = table_number
            sample_data['doi'] = paper['doi']
            sample_data['sample'] = sample_name
            sample_data['source'] = file_label
            samples.append(sample_data)

# Combine and save
pred_df = pd.DataFrame(samples)
all_df = pd.concat([df, pred_df], ignore_index=True)
all_df['source_order'] = all_df['source'].apply(lambda x: 0 if x == 'real' else 1)
all_df.sort_values(by=['table_number', 'sample', 'source_order', 'source'], inplace=True)
all_df.drop(columns='source_order', inplace=True)
all_df.to_csv('test_results.csv', index=False, encoding='utf-8-sig')  # Use utf-8-sig for
