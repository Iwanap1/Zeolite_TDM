import torch
from schema import Output, create_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
from outlines import generate
from outlines.models import transformers as outlines_transformers
import os
import json
import pandas as pd
from tabulate import tabulate


def load_model(name):
    model = outlines_transformers(
        name,
        torch_dtype=torch.float16,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(name)
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

def format_llama2_chat_prompt(messages):
    """
    Format messages using the LLaMA-2 chat template.
    """
    system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible."
    formatted = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"

    for i, message in enumerate(messages):
        role = message["role"]
        content = message["content"]
        if role == "user":
            if i > 0:
                formatted += "</s><s>[INST] "
            formatted += content + " [/INST]"
        elif role == "assistant":
            formatted += f" {content}"
    formatted += " </s>"
    return formatted


def main(data_format, model="meta-llama/Meta-Llama-3-8B-Instruct"):
    model, tokenizer = load_model(model)
    generator = generate.json(model, Output)

    file_path = os.path.join(os.environ["HOME"], "Zeolite_TDM/table_extraction/test_data.json")
    with open(file_path, "r") as f:
        data = json.load(f)

    results = []

    for test_table in data:
        if data_format != 'string':
            df = pd.DataFrame(test_table[data_format])
            markdown_table = tabulate(df.values.tolist(), tablefmt="github", showindex=False, headers=[])
            prompt = create_prompt(markdown_table)
        else:
            prompt = create_prompt(test_table["string"])

        messages = [
            {"role": "system", "content": "You are a helpful assistant that extracts structured zeolite data from tables."},
            {"role": "user", "content": prompt}
        ]
        print("⏳ Generating...")

        try:
            chat_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print("Could not apply chat template, constructing it manually")
            chat_prompt = tokenizer.tokenize(format_llama2_chat_prompt(messages))

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

    with open(os.path.join(os.environ["HOME"], f"Zeolite_TDM/table_extraction/test_results_{data_format}.json"), "w") as f:
        json.dump(results, f, indent=2)



# -----------------------------
# 5. Run
# -----------------------------
if __name__ == "__main__":
    main("string")
    main("table")
    main("flat_table")
    main("string", model='../models/llama2-7b-chat-hf-table')
