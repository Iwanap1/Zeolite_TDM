# Just runs over the full dataset again. Test papers can be found in o_log of fine-tuning

from dynamic_schemas import make_schema
import json
from outlines import models, generate
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import os
import datetime
from prepare_data import construct_process_identification_user_prompt, system
import time


def extract_zeolite_names_from_prompt(prompt: str):
    # for test data only
    match = re.search(r"Zeolites:\s*(.+?)(?:\n|$)", prompt)
    if not match:
        return []
    return [z.strip() for z in match.group(1).split(",")]


def load_model(model_path, use_outlines):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if use_outlines:
        model = models.transformers(model_path, device="cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
    return model, tokenizer


def zeolite_equal(pred, gold):
    return (
        pred.get("zeolite_source") == gold.get("zeolite_source") and
        (pred.get("post_synthesis") or []) == (gold.get("post_synthesis") or [])
    )


def test_with_outlines(file_path, model_path, output_path="../data/process_identification_results.json"):
    """
    Args: file_path is to jsonl, expected to already be in chat format: {'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}, {'role': 'assistant', 'content': assistant_prompt}]}
    """
    all_results = []
    model, tokenizer = load_model(model_path, use_outlines=True)

    with open(file_path, "r") as f:
        for line in f: 
            paper = json.loads(line)
            user_prompt = paper["messages"][1]["content"]
            zeolite_names = extract_zeolite_names_from_prompt(user_prompt)

            if not zeolite_names:
                print("⚠️  Skipping: No zeolite names found.")
                continue

            ZeoliteHistorySchema = make_schema(zeolite_names)
            generator = generate.json(model, ZeoliteHistorySchema)

            chat_input = tokenizer.apply_chat_template(
                paper["messages"],
                tokenize=False,
                add_generation_prompt=True
            )

            try:
                raw = generator(chat_input)
                print("🔹 Raw output:", raw)

                if isinstance(raw, tuple) and raw[0] == "root":
                    raw = raw[1]

                prediction_obj = ZeoliteHistorySchema.model_validate(raw)
                prediction = prediction_obj.model_dump()
                print("✅ Parsed:", prediction)

            except Exception as e:
                print(f"❌ Error during generation: {e}")
                prediction = raw if isinstance(raw, dict) else {"raw_output": str(raw)}
                all_results.append({
                    "doi": paper.get("doi", ""),
                    "text": user_prompt,
                    "predicted": prediction,
                    "true": paper["messages"][2]["content"],
                    "status": "invalid"  # or "correct", etc.
                })
                continue

            try:
                true = json.loads(paper["messages"][2]["content"])
            except Exception as e:
                print(f"⚠️  Invalid gold format: {e}")
                continue


            all_same = all(
                zeolite_equal(prediction.get(name, {}), true.get(name, {}))
                for name in zeolite_names
            )

            if all_same:
                status = "correct"
            elif any(
                prediction.get(name, {}).get("post_synthesis") and
                true.get(name, {}).get("post_synthesis") and
                prediction[name]["post_synthesis"][0] == true[name]["post_synthesis"][0]
                for name in zeolite_names
            ):
                status = "correct_start"
            else:
                status = "incorrect"

            all_results.append({
                "doi": paper.get("doi", ""),
                "text": user_prompt,
                "predicted": prediction,
                "true": true,
                "status": status
            })

    with open(output_path, "w") as out_f:
        json.dump(all_results, out_f, indent=2)
        
    return all_results


def test_without_outlines(file_path, model_path, output_path="../data/process_identification_results_no_outlines.json"):
    """
    Runs test without using outlines (raw generation with HuggingFace Transformers).
    Expects assistant message (index 2) to contain JSON string.
    """
    import torch
    from transformers import TextStreamer

    all_results = []
    model, tokenizer = load_model(model_path, use_outlines=False)
    model.eval()

    with open(file_path, "r") as f:
        for line in f:
            paper = json.loads(line)
            user_prompt = paper["messages"][1]["content"]
            zeolite_names = extract_zeolite_names_from_prompt(user_prompt)

            if not zeolite_names:
                print("⚠️ Skipping: No zeolite names found.")
                continue

            chat_input = tokenizer.apply_chat_template(
                paper["messages"],
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(chat_input, return_tensors="pt").to(model.device)

            try:
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                decoded = tokenizer.decode(output[0], skip_special_tokens=True)

                # Try to isolate JSON portion (use more robust parsing if needed)
                json_start = decoded.find("{")
                json_text = decoded[json_start:].strip()
                prediction = json.loads(json_text)

                print("✅ Parsed:", prediction)

            except Exception as e:
                print(f"❌ Error during generation: {e}")
                prediction = {"raw_output": decoded if 'decoded' in locals() else "N/A"}
                all_results.append({
                    "doi": paper.get("doi", ""),
                    "text": user_prompt,
                    "predicted": prediction,
                    "true": paper["messages"][2]["content"],
                    "status": "invalid"
                })
                continue

            try:
                true = json.loads(paper["messages"][2]["content"])
            except Exception as e:
                print(f"⚠️ Invalid gold format: {e}")
                continue

            all_same = all(
                zeolite_equal(prediction.get(name, {}), true.get(name, {}))
                for name in zeolite_names
            )

            if all_same:
                status = "correct"
            elif any(
                prediction.get(name, {}).get("post_synthesis") and
                true.get(name, {}).get("post_synthesis") and
                prediction[name]["post_synthesis"][0] == true[name]["post_synthesis"][0]
                for name in zeolite_names
            ):
                status = "correct_start"
            else:
                status = "incorrect"

            all_results.append({
                "doi": paper.get("doi", ""),
                "text": user_prompt,
                "predicted": prediction,
                "true": true,
                "status": status
            })

    with open(output_path, "w") as out_f:
        json.dump(all_results, out_f, indent=2)

    return all_results


def load_mongo(uri="mongodb://localhost:27017/", db_name="zeolite_tdm"):
    """
    Load MongoDB collections for papers and samples.
    Returns:
        tuple: MongoDB collections for papers and samples
    """
    import pymongo
    client = pymongo.MongoClient(uri)
    db = client[db_name]
    papers = db["papers"]
    samples = db["samples"]
    return papers, samples


def run_process_identification_pipe(
    model_path,
    mongo_uri="mongodb://localhost:27017/",
    mongo_db_name="zeolite_tdm",
    claim_size=5,
    max_runtime=300
):
    """
    Run the process identification pipeline.
    Args:
        model_path (str): Path to the model.
        test_data_path (str): Path to the test data file.
        output_path (str): Path to save the results.
        use_outlines (bool): Whether to use outlines for generation.
        mongo_uri (str): MongoDB URI.
        mongo_db_name (str): MongoDB database name.
        claim_size (int): Number of papers to claim for processing per iteration.
    """
    start_time = time.time()
    papers_collection, samples_collection = load_mongo(uri=mongo_uri, db_name=mongo_db_name)
    model, tokenizer = load_model(model_path, use_outlines=True)
    while True:
        now = time.time()
        if now - start_time > max_runtime:
            print("Time expired, exiting")
            break

        claimed = claim_papers(papers_collection, claim_size)

        if not claimed:
            print("No more papers to process.")
            break
        
        for paper in claimed:
            paper_samples = get_samples_for_paper(samples_collection, paper["_id"])
            if not paper_samples:
                papers_collection.find_one_and_update(
                    {"_id": paper["_id"]},
                    {"$set": {"status": "rejected", 'rejected_because': 'could not find any samples for process identification'}}
                )
                continue
            samples_names = [paper_sample["name"] for paper_sample in paper_samples]
            try:
                paper_extract = paper["extract"]
            except KeyError:
                papers_collection.find_one_and_update(
                    {"_id": paper["_id"]},
                    {"$set": {"status": "rejected", 'rejected_because': 'no text extract found for process identification'}}
                )
                continue

            user_prompt = construct_process_identification_user_prompt(samples_names, paper_extract)
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ]

            ZeoliteProcesses = make_schema(samples_names)
            generator = generate.json(model, ZeoliteProcesses)

            try:
                chat_input = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                raw = generator(chat_input)
                save_samples(raw, samples_collection)
                papers_collection.find_one_and_update(
                    {"_id": paper["_id"]},
                    {
                        "$set": {
                            "status": "awaiting final extraction"
                        },
                        "$unset": {
                            "process_claimed_at": ""
                        }
                    }
                )
            except Exception as e:
                print(f"Error during generation: {e}")
                print('paper:', paper.get('doi', ''))
                papers_collection.find_one_and_update(
                    {"_id": paper["_id"]},
                    {
                        "$set": {
                            "status": "rejected",
                            "rejected_because": "error during process identification"
                        },
                        "$unset": {
                            "process_claimed_at": ""
                        }
                    }
                )
                continue
        

def claim_papers(papers, n):
    paper_ids = []
    for paper in papers.find({"status": "awaiting process identification"}).limit(n):
        res = papers.update_one(
            {"_id": paper["_id"], "status": "awaiting process identification", "process_claimed_at": {"$exists": False}},
            {"$set": {"status": "processing", "process_claimed_at": datetime.utcnow()}}
        )
        if res.modified_count > 0:
            paper_ids.append(paper["_id"])
    return list(papers.find({"_id": {"$in": paper_ids}}))


def get_samples_for_paper(samples_collection, paper_id):
    return list(samples_collection.find({"paper_id": paper_id}))


def save_samples(raw_output, samples_collection):
    for name, content in raw_output.items():
        sample = samples_collection.find_one({"name": name})
        if sample:
            samples_collection.update_one(
                {"_id": sample["_id"]},
                {"$set": {
                    "post_synthesis_steps": content.get("post_synthesis"),
                    "zeolite_source_step": content.get("zeolite_source"),
                    "morphological_description": content.get("morphological_description")
                }}
            )
