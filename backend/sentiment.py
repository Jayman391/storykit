from collections import defaultdict
import shifterator as sh
import pandas as pd
import traceback
import numpy as np

# Precompute labmt_dict as a Pandas Series for faster lookups
labmt = pd.read_csv('data/labmt.csv')
labmt = labmt[['Word', 'Happiness Score']]
labmt_dict = labmt.set_index('Word')['Happiness Score']

def make_daily_sentiments(days: dict) -> dict:
    sentiments = {}
    
    # Convert labmt_dict index to a NumPy array for faster operations
    labmt_words = labmt_dict.index.to_numpy()
    
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


def make_daily_wordshifts(days: dict, window: int = 2):
    """
    Compute daily wordshifts and sentiments.
    """
    shifts = []
    index = 0
    daykeys = list(days.keys())

    for i, day in days.items():
        if index > 0:
            day_words = day.get('1-gram', {}).get('counts', {})
            # Filter words present in labmt
            day_words = {k: v for k, v in day_words.items() if k in labmt_dict} 
            day_counts = sum(day_words.values())
            if day_counts == 0:
                index += 1
                continue
            # Calculate sentiment for each word
            day_sentiment = {k: (v/day_counts) * labmt_dict[k] for k, v in day_words.items()}
            total_day_sentiment = sum(day_sentiment.values())
            prev_days = daykeys[max(0, index-window):index]
            index += 1

            prev_words = defaultdict(int)

            for prev_day in prev_days:
                prev_day_words = days.get(prev_day, {}).get('1-gram', {}).get('counts', {})
                prev_day_word_freqs = {k: v for k, v in prev_day_words.items() if k in labmt_dict}
                for k, v in prev_day_word_freqs.items():
                    prev_words[k] += v
                
            prev_words = dict(prev_words)

            if day_words and prev_words:
                try:
                    shift = sh.WeightedAvgShift(
                        type2freq_1=day_words, 
                        type2freq_2=prev_words, 
                        type2score_1='labMT_English',
                        type2score_2='labMT_English',
                        reference_value=total_day_sentiment, 
                        handle_missing_scores='exclude'
                    ).get_shift_graph()
                    
                    shifts.append(shift)
                except Exception as e:
                    print(f"Error generating shift graph for day {i}: {e}")
                    traceback.print_exc()
        else:
            index += 1

    return shifts
