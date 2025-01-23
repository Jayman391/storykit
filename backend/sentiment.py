from collections import defaultdict
import shifterator as sh
import pandas as pd
import traceback


labmt = pd.read_csv('data/labmt.csv')
labmt = labmt[['Word','Happiness Score']]
# Ensure that make_daily_wordshifts_and_sentiments is defined before its usage
def make_daily_wordshifts_and_sentiments(days: dict, window: int = 2):
    """
    Compute daily wordshifts and sentiments.
    """
    # Load the labMT data
    labmt = pd.read_csv('data/labmt.csv')
    labmt = labmt[['Word','Happiness Score']]
    # Create a mapping from Word to Happiness Score
    labmt_dict = labmt.set_index('Word')['Happiness Score'].to_dict()
    
    sentiments = {}
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
                sentiments[i] = 0
                index += 1
                continue
            # Calculate sentiment for each word
            day_sentiment = {k: (v/day_counts) * labmt_dict[k] for k, v in day_words.items()}
            total_day_sentiment = sum(day_sentiment.values())
            sentiments[i] = total_day_sentiment
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

    return sentiments, shifts
