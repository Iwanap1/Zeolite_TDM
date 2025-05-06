import torch
from schema import Output, create_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
from outlines import generate
from outlines.models import transformers as outlines_transformers
import os
import json
import pandas as pd
from tabulate import tabulate


def load_model():
    model = outlines_transformers(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    return model, tokenizer


def evaluate_accuracy(real_samples, pred_samples):
    total_fields = 0
    correct_fields = 0
    unmatched_samples = 0

    real_by_name = {s["sample"]: s for s in real_samples}
    pred_by_name = {s["sample"]: s for s in pred_samples}

    for sample_name, real_sample in real_by_name.items():
        pred_sample = pred_by_name.get(sample_name)
        if not pred_sample:
            unmatched_samples += 1
            continue

        for field, real_value in real_sample.items():
            if field == "sample":
                continue  # skip sample name
            total_fields += 1
            pred_value = pred_sample.get(field)

            if pred_value is None:
                continue

            # Loose match: allow extra info like units or measurement method
            if real_value.strip() in pred_value.strip():
                correct_fields += 1

    return {
        "correct_fields": correct_fields,
        "total_fields": total_fields,
        "accuracy": round((correct_fields / total_fields) * 100, 2) if total_fields else 0.0,
        "unmatched_samples": unmatched_samples
    }


def main():
    model, tokenizer = load_model()
    generator = generate.json(model, Output)

    file_path = os.path.join(os.environ["HOME"], "tdm/table_extraction/data.json")
    with open(file_path, "r") as f:
        data = json.load(f)

    results = []

    for test_table in data:
        df = pd.DataFrame(test_table['flat_table'])
        markdown_table = tabulate(df.values.tolist(), tablefmt="github", showindex=False, headers=[])
        prompt = create_prompt(markdown_table)

        print("⏳ Generating...")
        chat_prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant that extracts structured zeolite data from tables."},
            {"role": "user", "content": prompt}
        ],
        tokenize=False,
        add_generation_prompt=True
    )
        output = generator(chat_prompt)
        pred = output.model_dump(exclude_none=True)["samples"]

        eval = evaluate_accuracy(
            real_samples=test_table.get("samples", []),
            pred_samples=pred
        )
        
        results.append({
            "doi": test_table.get("doi", "unknown"),
            "real": test_table.get("samples"),
            "pred": pred,
            "evaluation": eval
        })

    with open(os.path.join(os.environ["HOME"], "tdm/table_extraction/test_results.json"), "w") as f:
        json.dump(results, f, indent=2)



# -----------------------------
# 5. Run
# -----------------------------
if __name__ == "__main__":
    main()
