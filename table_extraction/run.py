import torch
from schema import Output, create_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
from outlines import generate
from outlines.models import transformers as outlines_transformers
import os
import pandas as pd
from tabulate import tabulate
import time
import uuid
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")


def load_mongo():
    db = MongoClient(os.environ["MONGO_URI"])["zeolite_tdm"]
    papers = db["papers"]
    tables = db["tables"]
    samples = db["samples"]
    return papers, tables, samples


def classify_tables(tables):
    update = tables.update_many({
        "$and": [
            {"contains_bet": {'$exists': False}},
            {
                "$or": [
                    {'caption': {'$regex': 'n2 adsorption', '$options': 'i'}},
                    {'caption': {'$regex': 'surface area', '$options': 'i'}},
                    {'caption': {'$regex': 'pore volume', '$options': 'i'}},
                    
                    {'as_string': {'$elemMatch': {'$regex': 'sbet', '$options': 'i'}}},
                    {'as_string': {'$elemMatch': {'$regex': 'vmicro', '$options': 'i'}}},
                    {'as_string': {'$elemMatch': {'$regex': 'Sext'}}},
                    {'as_string': {'$elemMatch': {'$regex': 'pore volume', '$options': 'i'}}},
                    {'as_string': {'$elemMatch': {'$regex': 'surface area', '$options': 'i'}}},
                    {'as_string': {'$elemMatch': {'$regex': 'm2/g', '$options': 'i'}}},
                    {'as_string': {'$elemMatch': {'$regex': 'cm3/g', '$options': 'i'}}},
                    {
                        "$and": [
                            {'as_string': {'$elemMatch': {'$regex': 'cm3', '$options': 'i'}}},
                            {'as_string': {'$elemMatch': {'$regex': 'g-1', '$options': 'i'}}}
                        ]
                    },                
                    {
                        "$and": [
                            {'as_string': {'$elemMatch': {'$regex': 'm2', '$options': 'i'}}},
                            {'as_string': {'$elemMatch': {'$regex': 'g-1', '$options': 'i'}}}
                        ]
                    },
                ]
            }
        ]
    },
    {"$set": {"contains_bet": True}}
    )
    print(update)


def reject_papers_without_bet_table():
    papers, tables, paras = load_mongo()

    # Find paper IDs that have any BET table
    papers_with_bet = tables.distinct("paper_id", {"contains_bet": True})

    # Now find papers that are awaiting table extraction but don't have any BET tables
    rejected_papers = papers.find({
        "status": "awaiting table extraction",
        "_id": {"$nin": papers_with_bet}
    })

    rejected_count = 0

    for paper in rejected_papers:
        paper_id = paper["_id"]
        doi = paper.get("doi", "[no DOI]")

        papers.update_one({"_id": paper_id}, {
            "$set": {
                "status": "rejected",
                "rejected_because": "no BET table"
            }
        })
        tables.delete_many({"paper_id": paper_id})
        paras.delete_many({"paper_id": paper_id})
        print(f"🚫 Rejected paper {doi} (no BET table)")

        rejected_count += 1

    print(f"✅ Rejected {rejected_count} papers due to missing BET tables.")


def load_model():
    model = outlines_transformers(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    return model, tokenizer



def main(max_minutes=1000, batch_size=1):
    model, tokenizer = load_model()
    generator = generate.json(model, Output)
    papers, tables, samples = load_mongo()
    worker_id = str(uuid.uuid4())

    print(f"🧠 Worker ID: {worker_id}")
    start_time = time.time()
    end_time = start_time + max_minutes * 60

    while time.time() < end_time:
        target_paper_ids = papers.find({"status": "awaiting table extraction"}).distinct("_id")
        if not target_paper_ids:
            print("✅ All tables processed.")
            break

        claimed_ids = []
        for doc in tables.find({
            "paper_id": {"$in": target_paper_ids},
            "contains_bet": True,
            "claimed": {"$exists": False},
            "extracted": {"$exists": False}
        }).limit(batch_size * 10):

            result = tables.update_one(
                {"_id": doc["_id"], "claimed": {"$exists": False}},
                {"$set": {"claimed": worker_id}}
            )
            if result.modified_count == 1:
                claimed_ids.append(doc["_id"])

        if not claimed_ids:
            print("⏱️ No more claimable tables at this moment.")
            time.sleep(5)
            continue

        now = time.time()
        if now >= end_time:
            print("⏱️ Time expired, processing final claimed batch before exit.")

        table_data = list(tables.find({"_id": {"$in": claimed_ids}}))

        for table_dict in table_data:
            try:
                df = pd.DataFrame(table_dict['single_head_table'])
                markdown_table = tabulate(df.values.tolist(), tablefmt="github", showindex=False, headers=[])
                prompt = create_prompt(markdown_table)

                print("⏳ Generating sample from table...")
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

                docs = []
                for row in pred:
                    doc = {
                        "paper_id": table_dict["paper_id"],
                        "timestamp": time.time(),
                        **row  # Unpack each field from the sample row directly
                    }
                    docs.append(doc)
                if docs:
                    samples.insert_many(docs)

                tables.update_one({"_id": table_dict["_id"]}, {"$set": {"extracted": True}})
                print(f"✅ Extracted table: {table_dict.get('label') or '[no label]'}")

            except Exception as e:
                print(f"⚠️ Failed on table {table_dict.get('_id')}: {e}")

        # Check for papers that are done with all their BET tables
        for paper_id in target_paper_ids:
            unfinished = tables.count_documents({
                "paper_id": paper_id,
                "contains_bet": True,
                "extracted": {"$exists": False}
            })
            if unfinished == 0:
                papers.update_one({"_id": paper_id}, {"$set": {"status": "awaiting process identification"}})
                print(f"Completed extraction for paper {paper_id}")
                deleted = tables.delete_many({"paper_id": paper_id}).deleted_count
                print(f"Deleted {deleted} tables for paper {paper_id}")


if __name__ == "__main__":
    papers, tables, samples = load_mongo()
    classify_tables(tables)
    reject_papers_without_bet_table()
    main(max_minutes=1000, batch_size=1, papers=papers, tables=tables, samples=samples)
