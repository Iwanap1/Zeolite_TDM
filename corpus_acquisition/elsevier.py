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

