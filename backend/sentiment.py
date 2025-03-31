from io import BytesIO
import base64
import shifterator as sh
import pandas as pd
import traceback
import numpy as np
from multiprocessing import Pool, cpu_count
from collections import Counter
import matplotlib.pyplot as plt


def compute_single_day_sentiment(args):
    """
    Compute sentiment for a single day using direct dictionary lookups.

    Unpacked arguments:
      day_id: the key (ID) for the day.
      day_data: the dictionary that holds 1-gram counts.
      labmt_scores: a dictionary mapping valid words to their happiness scores.
    """
    day_id, day_data, labmt_scores = args

    # Retrieve the 1-gram counts; if missing, skip this day.
    day_words = day_data.get('1-gram', {}).get('counts', {})
    if not day_words:
        return day_id, None

    total_counts = 0.0
    weighted_sentiment = 0.0

    # Iterate over words and counts, accumulating weighted scores.
    for word, count in day_words.items():
        score = labmt_scores.get(word)
        if score is not None:
            total_counts += count
            weighted_sentiment += count * score

    if total_counts == 0:
        return day_id, None

    # Return the weighted average sentiment.
    return day_id, weighted_sentiment / total_counts

def make_daily_sentiments_parallel(days: dict, labmt_dict, smoothing: int = 1) -> dict:
    """
    Compute daily sentiments in parallel. Returns a dictionary of {day_id: sentiment}.

    Parameters:
      days: A dictionary where keys are day IDs and values are day_data dictionaries.
      labmt_dict: A Pandas Series with words as the index and happiness scores as values.
      smoothing: The window size for smoothing sentiment values (if > 1).
    """
    # Convert the labMT Pandas Series into a dictionary for fast lookups.
    labmt_scores = labmt_dict

    # Prepare the tasks for parallel processing.
    tasks = [(day_id, day_data, labmt_scores) for day_id, day_data in days.items()]

    results_dict = {}
    with Pool(processes=cpu_count()) as pool:
        for day_id, sentiment in pool.map(compute_single_day_sentiment, tasks):
            if sentiment is not None:
                results_dict[day_id] = sentiment

    # If smoothing is requested, perform a simple moving average.
    if smoothing > 1:
        smoothed_results = {}
        # Convert to lists to avoid repeated conversion in the loop.
        day_ids = list(results_dict.keys())
        sentiments = list(results_dict.values())
        for idx in range(smoothing, len(sentiments)):
            smoothed_results[day_ids[idx]] = np.mean(sentiments[idx - smoothing:idx])
        return smoothed_results

    return results_dict

def compute_shift(args):
    day, day_words, window_words, total_day_sentiment = args
    try:
        fig, axs = plt.subplots(1, 1, figsize=(10, 20))

        sh.WeightedAvgShift(
            type2freq_1=day_words,
            type2freq_2=window_words,
            type2score_1='labMT_English',
            type2score_2='labMT_English',
            reference_value=total_day_sentiment,
            handle_missing_scores='exclude'
        ).get_shift_graph(axs)

        fig.tight_layout()
        # Convert plot to base64 image
        buf = BytesIO()
        fig.savefig(buf, format='svg')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        encodedstr = f"data:image/svg+xml;base64,{encoded}"

       # Ensure valid range
        return day, encodedstr
    except Exception as e:
        print(f"Error generating shift graph for day {day}: {e}")
        traceback.print_exc()
        return None

def make_daily_wordshifts_parallel(days: dict, labmt_dict, window: int = 2):
    shifts = []
    daykeys = list(days.keys())
    labmt_keys = set(labmt_dict.keys())

    # Pre-filter days
    filtered_days = {}
    for day in daykeys:
        counts = days[day].get('1-gram', {}).get('counts', {})
        filtered_counts = {k: v for k, v in counts.items() if k in labmt_keys}
        if filtered_counts:
            filtered_days[day] = filtered_counts

    window_counter = Counter()
    window_days = []
    tasks = []

    for idx, day in enumerate(daykeys):
        if day not in filtered_days:
            continue  # Skip days with no relevant words

        day_words = filtered_days[day]
        day_counts = sum(day_words.values())
        if day_counts == 0:
            continue

        # Calculate sentiment
        day_sentiment = sum((v / day_counts) * labmt_dict[k] for k, v in day_words.items())

        # Update sliding window
        if len(window_days) == window + 1:
            # Remove the oldest day from the window
            oldest_day = window_days.pop(0)
            window_counter.subtract(filtered_days[oldest_day])
            window_counter += Counter()

        # Add the current day to the window
        window_days.append(day)
        window_counter.update(day_words)

        if not window_counter:
            continue

        # Prepare task for parallel processing
        tasks.append((day, day_words, dict(window_counter), day_sentiment))

    # Use multiprocessing Pool
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(compute_shift, tasks)

    # Filter out None results due to exceptions
    shifts = [shift for shift in results if shift is not None]

    return shifts