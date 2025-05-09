import os
import requests
from pathlib import Path
from math import ceil

# Constants
OUTPUT_DIR = Path("data/xml")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PAPERS_PER_BATCH = 25  # Scopus API returns up to 25 per request

def extract_articles(search_results: list, api_key: str, max_articles: int) -> tuple[int, int]:
    """
    Extracts article XMLs from Scopus API results and saves them to disk.

    Args:
        search_results (list): List of JSON responses from Scopus API containing article metadata.
        api_key (str): Your Elsevier API key.

    Returns:
        tuple[int, int]: A tuple containing:
            - Number of successful downloads
            - Number of failed downloads
    """
    success_count = 0
    failure_count = 0

    for batch in search_results:
        for entry in batch.get('search-results', {}).get('entry', []):
            if success_count >= max_articles:
                return success_count, failure_count

            pii = entry.get('pii')
            if not pii:
                failure_count += 1
                continue

            url = f'https://api.elsevier.com/content/article/pii/{pii}'
            headers = {
                "X-ELS-APIKey": api_key,
                "Accept": "text/xml"
            }

            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()

                output_path = OUTPUT_DIR / f"{pii}.xml"
                with open(output_path, 'w', encoding='utf-8') as file:
                    file.write(response.text)

                success_count += 1

            except requests.RequestException as e:
                failure_count += 1
                print(f"Failed to download {pii}: {e}")

    return success_count, failure_count

def collect_elsevier_papers(
    query: str,
    paper_count: int,
    api_key: str,
    start_batch: int = 0
) -> None:
    """
    Fetches and saves Elsevier article XMLs from the Scopus API.

    Args:
        query (str): A Scopus search query (e.g., "machine learning").
        paper_count (int): Total number of papers to fetch (will determine batch count).
        api_key (str): Your Elsevier API key.
        start_batch (int, optional): Starting offset for Scopus pagination. Default is 0.

    Output:
        Prints a summary of success/failure after processing.
    """
    url = 'https://api.elsevier.com/content/search/scopus'
    results = []
    batch_count = ceil(paper_count / PAPERS_PER_BATCH)

    for i in range(batch_count):
        start = start_batch + i * PAPERS_PER_BATCH
        params = {
            'query': query,
            'count': PAPERS_PER_BATCH,
            'start': start
        }
        headers = {
            "X-ELS-APIKey": api_key,
            "Accept": "application/json"
        }

        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            results.append(response.json())
        except requests.RequestException:
            continue  # Skip failed batch

    print("Collecting papers...")
    success, failure = extract_articles(results, api_key, max_articles=paper_count)
    print(f"Successfully collected {success} papers.")
    print(f"Failed to collect {failure} papers.")


def collect_elsevier_to_mongo(
    query: str,
    max_minutes: int,
    api_key: str,
    mongo_uri: str = "mongodb://localhost:27017/",
    db_name: str = "zeolite_tdm"
) -> None:
    import time
    import requests
    from pymongo import MongoClient, InsertOne
    from elsevier_parser import ElsevierParser

    PAPERS_PER_BATCH = 25  # You might want to configure this at the top level
    client = MongoClient(mongo_uri)
    db = client[db_name]
    parser = ElsevierParser()

    latest_batch = db.papers.find_one(
        {"batch": {"$exists": True}},
        sort=[("batch", -1)],
        projection={"batch": 1}
    )
    batch_number = (latest_batch["batch"] + 1) if latest_batch else 0

    print(f"Starting from batch {batch_number}...")

    start_time = time.time()
    end_time = start_time + (max_minutes * 60)
    total_success = 0
    total_failure = 0

    while time.time() < end_time:
        start = batch_number * PAPERS_PER_BATCH
        params = {
            'query': query,
            'count': PAPERS_PER_BATCH,
            'start': start,
            'field': 'pii'
        }
        headers = {
            "X-ELS-APIKey": api_key,
            "Accept": "application/json"
        }

        try:
            response = requests.get('https://api.elsevier.com/content/search/scopus', params=params, headers=headers)
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as e:
            print(f"Batch {batch_number} failed: {e}")
            time.sleep(1)
            batch_number += 1
            continue

        entries = result.get('search-results', {}).get('entry', [])
        if not entries:
            print("No more entries returned. Ending early.")
            break

        paper_ops = []
        paragraph_docs = []
        table_docs = []

        for entry in entries:
            if time.time() >= end_time:
                print("Time limit reached, stopping.")
                break

            pii = entry.get('pii')
            if not pii:
                total_failure += 1
                continue

            article_url = f'https://api.elsevier.com/content/article/pii/{pii}'
            try:
                article_resp = requests.get(article_url, headers={"X-ELS-APIKey": api_key, "Accept": "text/xml"})
                article_resp.raise_for_status()

                parsed = parser.parse_from_string(article_resp.text, file_path=f"pii:{pii}")
                if parsed is None or not parsed.get("doi"):
                    total_failure += 1
                    continue

                # Skip if already exists (optional pre-check)
                if db.papers.count_documents({'doi': parsed["doi"]}, limit=1):
                    print(f"Skipping duplicate DOI: {parsed['doi']}")
                    continue

                paper_doc = {
                    "doi": parsed.get("doi"),
                    "batch": batch_number,
                }

                if parsed['rejected_because'] != 'accepted':
                    paper_doc.update({
                        "status": "rejected",
                        "rejected_because": parsed.get("rejected_because")
                    })
                else:
                    paper_doc.update({
                        **{k: parsed.get(k) for k in (
                            'file_path', 'type', 'title', 'authors', 'year',
                            'publication', 'keywords', 'abstract', 'rejected_because'
                        )},
                        "status": "awaiting paragraph classification"
                    })

                paper_ops.append(InsertOne(paper_doc))

                if parsed['rejected_because'] == 'accepted':
                    if parsed.get('sections'):
                        paragraph_docs.extend([
                            {"doi": parsed["doi"], "section": sec["section_name"], "text": para}
                            for sec in parsed["sections"]
                            for para in sec["content"]
                        ])
                    if parsed.get('tables'):
                        table_docs.extend([
                            {"doi": parsed["doi"], **table}
                            for table in parsed["tables"]
                        ])

                total_success += 1

            except Exception as e:
                total_failure += 1
                print(f"Error processing PII {pii}: {e}")

        # Insert in bulk
        if paper_ops:
            try:
                db.papers.bulk_write(paper_ops, ordered=False)
            except Exception as e:
                print(f"Error during bulk insert to papers: {e}")

        if paragraph_docs:
            try:
                db.paragraphs.insert_many(paragraph_docs, ordered=False)
            except Exception as e:
                print(f"Error during insert_many to paragraphs: {e}")

        if table_docs:
            try:
                db.tables.insert_many(table_docs, ordered=False)
            except Exception as e:
                print(f"Error during insert_many to tables: {e}")

        batch_number += 1

    print(f"\nInserted {total_success} papers.")
    print(f"Failed on {total_failure} papers.")

