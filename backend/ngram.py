from datetime import datetime
from collections import Counter
import re
import numpy as np

def compute_ngrams(data: list, params: dict) -> dict:
    dates = {}
    full_corpus = {}

    ngrams = params.get("keywords", [])
    if "all" in ngrams:    
        ns = [1]
        for ngram in ngrams:
            length = len(ngram.split())
            if length > 1:
                ns.append(length)
        ngrams.remove("all")
        if not ngrams:
            ngrams = []
        ns = np.unique(ns)
    else:
        ns = [params.get("n", 1)]
    # Initialize full corpus counters gifor each n
    for n in ns:
        full_corpus[f'{n}-gram'] = Counter()

    for doc in data:
        # Ensure date is a datetime object
        try:
            if isinstance(doc["date"], str):
                date_obj = datetime.strptime(doc["date"], '%Y-%m-%dT%H:%M:%S')
            else:
                date_obj = doc["date"]
            date_str = date_obj.strftime('%a, %d %b %Y %H:%M:%S')
        except Exception as e:
            print(f"Date parsing error: {e}")
            continue

        if date_str not in dates:
            dates[date_str] = {}
            for n in ns:
                dates[date_str][f'{n}-gram'] = Counter()

        # Combine 'text' and 'title' if 'title' exists
        text = doc.get("text", "")
        if 'title' in doc and doc['title']:
            text += ' ' + doc['title']
        text = text.lower()

        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        for n in ns:
            chunks = chunk_text(text, n)
            for chunk in chunks:
                if ngrams and chunk in ngrams:
                    dates[date_str][f'{n}-gram'][chunk] += 1
                    full_corpus[f'{n}-gram'][chunk] += 1
                elif not ngrams:
                    dates[date_str][f'{n}-gram'][chunk] += 1
                    full_corpus[f'{n}-gram'][chunk] += 1

    # Compute ranks for each date
    for date_str in dates:
        for n in dates[date_str]:
            counts = dates[date_str][n]
            sorted_ngrams = counts.most_common()
            ranks = {}
            current_rank = 1
            previous_count = None
            for idx, (ngram, count) in enumerate(sorted_ngrams):
                if count != previous_count:
                    current_rank = idx + 1
                    previous_count = count
                ranks[ngram] = current_rank
            dates[date_str][n] = {
                'counts': dict(counts),
                'ranks': ranks
            }

    # delete ngrams with counts less than 3
    for date_str in dates:
        for n in dates[date_str]:
            counts = dates[date_str][n]
            for ngram, count in list(counts['counts'].items()):
                if count < 3:
                    del counts['counts'][ngram]

    # Compute ranks for the full corpus
    for n in full_corpus:
        counts = full_corpus[n]
        sorted_ngrams = counts.most_common()
        ranks = {}
        current_rank = 1
        previous_count = None
        for idx, (ngram, count) in enumerate(sorted_ngrams):
            if count != previous_count:
                current_rank = idx + 1
                previous_count = count
            ranks[ngram] = current_rank
        full_corpus[n] = {
            'counts': dict(counts),
            'ranks': ranks
        }

    # delete ngrams with counts less than 3
    for n in full_corpus:
        counts = full_corpus[n]
        for ngram, count in list(counts['counts'].items()):
                if count < 3:
                    del counts['counts'][ngram]

    return {'dates': dates, 'full_corpus': full_corpus}

def chunk_text(text: str, n: int = 1):
    words = text.split()
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
