from collections import defaultdict
import shifterator as sh
import pandas as pd
import traceback
import numpy as np
from multiprocessing import Pool, cpu_count
from collections import Counter

# Precompute labmt_dict as a Pandas Series for faster lookups
labmt = pd.read_csv('data/labmt.csv')
labmt = labmt[['Word', 'Happiness Score']]
labmt_dict = labmt.set_index('Word')['Happiness Score']
labmt_words = labmt_dict.index.to_numpy()


def compute_single_day_sentiment(args):
    """
    Helper function to compute sentiment for a single day.
    Unpacked arguments:
      day_id: the key (ID) for the day
      day_data: the dictionary that holds 1-gram counts
      labmt_words: NumPy array of valid words (from the labMT dictionary)
      labmt_dict: Pandas Series for quick lookups of happiness scores
    """
    day_id, day_data, labmt_words, labmt_dict = args
    
    # Extract '1-gram' counts; default to empty dict if not present
    day_words_dict = day_data.get('1-gram', {}).get('counts', {})
    if not day_words_dict:
        return day_id, None  # Skip if no words for the day
    
    # Convert day_words_dict to two separate NumPy arrays
    words, counts = zip(*day_words_dict.items())
    words = np.array(words, dtype=object)
    counts = np.array(counts, dtype=np.float64)  # Ensure float for division
    
    # Create a mask for words present in labmt_dict
    mask = np.isin(words, labmt_words)
    if not np.any(mask):
        return day_id, None
    
    # Filter words and counts
    valid_words = words[mask]
    valid_counts = counts[mask]
    
    # Retrieve corresponding happiness scores using Pandas' indexing
    valid_scores = labmt_dict.loc[valid_words].to_numpy()
    
    # Calculate total counts to normalize
    total_counts = valid_counts.sum()
    if total_counts == 0:
        return day_id, None
    
    # Compute sentiment: (count / total_counts) * happiness_score
    sentiments_array = (valid_counts / total_counts) * valid_scores
    total_day_sentiment = sentiments_array.sum()
    
    return day_id, total_day_sentiment

def make_daily_sentiments_parallel(days: dict) -> dict:
    """
    A parallelized version of make_daily_sentiments. Returns a dictionary
    of {day_id: sentiment}.
    """
    labmt_words = labmt_dict.index.to_numpy()
    
    # Prepare arguments for each day
    tasks = [(day_id, day_data, labmt_words, labmt_dict)
             for day_id, day_data in days.items()]
    
    results_dict = {}
    with Pool(processes=cpu_count()) as pool:
        for day_id, sentiment in pool.map(compute_single_day_sentiment, tasks):
            if sentiment is not None:
                results_dict[day_id] = sentiment
    
    return results_dict

def compute_shift(args):
    day, day_words, window_words, total_day_sentiment = args
    try:
        shift = sh.WeightedAvgShift(
            type2freq_1=day_words,
            type2freq_2=window_words,
            type2score_1='labMT_English',
            type2score_2='labMT_English',
            reference_value=total_day_sentiment,
            handle_missing_scores='exclude'
        ).get_shift_graph()
        return shift
    except Exception as e:
        print(f"Error generating shift graph for day {day}: {e}")
        traceback.print_exc()
        return None

def make_daily_wordshifts_parallel(days: dict, window: int = 2):
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
        if len(window_days) == window:
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

def generate_wordshift_for_date(day, days):
    """
    Generate a wordshift graph for a specific day.
    """
    try:
        day_data = days.get(day, {})
        day_words = day_data.get('1-gram', {}).get('counts', {})
        if not day_words:
            return None

        # Filter words present in labmt_dict
        filtered_words = {k: v for k, v in day_words.items() if k in labmt_dict}
        if not filtered_words:
            return None

        total_day_counts = sum(filtered_words.values())
        if total_day_counts == 0:
            return None

        # Calculate sentiment
        day_sentiment = sum((v / total_day_counts) * labmt_dict[k] for k, v in filtered_words.items())

        # For wordshift, you might want to define a window or use specific comparison
        # Here, we'll compare the day's word distribution to the overall distribution
        overall_freq = Counter()
        for d, data in days.items():
            counts = data.get('1-gram', {}).get('counts', {})
            filtered = {k: v for k, v in counts.items() if k in labmt_dict}
            overall_freq.update(filtered)
        overall_freq = dict(overall_freq)

        shift = sh.WeightedAvgShift(
            type2freq_1=filtered_words,
            type2freq_2=overall_freq,
            type2score_1='labMT_English',
            type2score_2='labMT_English',
            reference_value=day_sentiment,
            handle_missing_scores='exclude'
        ).get_shift_graph()

        return shift

    except Exception as e:
        print(f"Error generating wordshift for {day}: {e}")
        traceback.print_exc()
        return None
