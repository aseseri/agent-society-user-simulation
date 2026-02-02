import numpy as np
import re
from collections import Counter


def analyze_user_pattern(user_profile):
    """
    Analyzes user history to enforce stylistic constraints (Length, Vocabulary).
    
    This module strictly enforces *style* but intentionally avoids biasing 
    the *star rating*. Experiments showed that enforcing statistical rating 
    averages caused a regression in accuracy (MAE) by overriding the 
    LLM's ability to detect nuanced sentiment in the raw history.
    """
    # Input validation
    if not isinstance(user_profile, dict): 
        return ""
    
    history = user_profile.get('history', [])
    if not history: 
        return ""


    # Pre-processing: Filter for valid, non-empty textual content
    texts = []
    for r in history:
        if r.get('text'):
            txt = str(r['text']).strip()
            if len(txt) > 5: # Threshold > 5 filters out low-signal/noise entries
                texts.append(txt)
            
    if not texts: 
        return ""


    # Verbosity Analysis
    # Compute average word count to determine length constraints
    word_counts = [len(t.split()) for t in texts]
    avg_len = np.mean(word_counts)

    if avg_len < 20:
        style_instr = "MANDATORY STYLE: Write an extremely short review (1-2 sentences)."
    elif avg_len > 120:
        style_instr = "MANDATORY STYLE: Write a long, detailed story (100+ words)."
    else:
        style_instr = "STYLE: Write a standard length review (3-5 sentences)."

    # Vocabulary Feature Extraction
    # Identify recurring signature words to mimic user voice
    all_text = " ".join(texts).lower()
    words = re.findall(r'\b[a-z]{4,}\b', all_text)

    # Domain-specific stop words to exclude from feature extraction
    common_stops = {'that', 'this', 'with', 'have', 'from', 'food', 'place', 'good', 'great', 'service', 'back', 'time', 'were', 'just', 'really', 'there', 'they', 'best', 'delicious', 'like', 'went', 'also', 'came', 'order'}
    meaningful = [w for w in words if w not in common_stops]
    
    vocab_instr = ""
    if meaningful:
        # Extract top 3 tokens that appear in multiple reviews
        top_words = Counter(meaningful).most_common(3)
        vocab_list = [w[0] for w, c in top_words if c >= 2]
        if vocab_list:
            vocab_instr = f"MANDATORY VOCABULARY: You must use these words: {', '.join(vocab_list)}."


    # Prompt Construction
    return (
        f"\n[SYSTEM: WRITING STYLE ENFORCEMENT]\n"
        f"1. {style_instr}\n"
        f"2. {vocab_instr}\n"
        f"INSTRUCTION: Ignore rating patterns. Focus purely on mimicking this writing style."
    )
