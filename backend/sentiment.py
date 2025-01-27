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

def make_daily_sentiments(days: dict) -> dict:
    sentiments = {}
    
    # Convert labmt_dict index to a NumPy array for faster operations
    
    for day_id, day_data in days.items():
        # Extract '1-gram' counts; default to empty dict if not present
        day_words_dict = day_data.get('1-gram', {}).get('counts', {})
        
        if not day_words_dict:
            continue  # Skip if no words for the day
        
        # Convert day_words_dict to two separate NumPy arrays
        words, counts = zip(*day_words_dict.items())
        words = np.array(words)
        counts = np.array(counts, dtype=np.float64)  # Ensure counts are float for division
        
        # Create a mask for words present in labmt_dict using NumPy's isin for speed
        mask = np.isin(words, labmt_words)
        
        if not np.any(mask):
            continue  # Skip if no relevant words
        
        # Filter words and counts based on the mask
        valid_words = words[mask]
        valid_counts = counts[mask]
        
        # Retrieve corresponding happiness scores using Pandas' indexing
        valid_scores = labmt_dict.loc[valid_words].to_numpy()
        
        # Calculate total counts to normalize
        total_counts = valid_counts.sum()
        
        if total_counts == 0:
            continue  # Avoid division by zero
        
        # Compute sentiment: (count / total_counts) * happiness_score
        sentiments_array = (valid_counts / total_counts) * valid_scores
        
        # Sum the sentiments to get the total sentiment for the day
        total_day_sentiment = sentiments_array.sum()
        
        # Assign to the sentiments dictionary
        sentiments[day_id] = total_day_sentiment
    
    return sentiments

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
